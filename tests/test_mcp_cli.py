"""MCP server (through the protocol, in-process) and CLI, offline."""

import asyncio
import json

import pytest

REF = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"
SUB = REF[:5] + "G" + REF[6:]
FASTA = "\n".join(f">{i}\n{s}" for i, s in
                  [("wt1", REF), ("wt2", REF), ("m1", SUB), ("m2", SUB), ("ins", REF[:20] + "WW" + REF[20:])])


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACE_RUNS_DIR", str(tmp_path))
    return tmp_path


def _payload(result):
    if getattr(result, "structured_content", None):
        sc = result.structured_content
        return sc.get("result", sc)
    return json.loads(result.content[0].text)


def test_mcp_tools_roundtrip(runs_dir):
    from mcp import Client

    from space.mcp_server import mcp

    async def go():
        async with Client(mcp) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert {"find_variants", "analyze_fasta", "get_variants", "get_variant", "annotate_run",
                    "map_run_to_structure", "list_runs"} <= names

            res = _payload(await client.call_tool("analyze_fasta", {"fasta": FASTA, "reference_sequence": REF,
                                                                    "label": "demo"}))
            run_id = res["run_id"]
            assert res["distinct_variants"] == 2
            assert {v["Mutation"] for v in res["top_variants"]} == {"I6G", "S20_R21insWW"}

            runs = _payload(await client.call_tool("list_runs", {}))
            assert run_id in {r["run_id"] for r in runs}

            v = _payload(await client.call_tool("get_variants", {"run_id": run_id, "min_frequency": 0.3}))
            assert [x["Mutation"] for x in v["variants"]] == ["I6G"]

            d = _payload(await client.call_tool("get_variant", {"run_id": run_id, "mutation": "i6g"}))
            assert sorted(d["carriers"]) == ["m1", "m2"]
            assert d["position"]["residue"] == "I"

            bad = await client.call_tool("get_variants", {"run_id": "../etc"})
            assert bad.is_error

            tsv = await client.read_resource(f"space://runs/{run_id}/variants")
            assert "I6G" in tsv.contents[0].text

    asyncio.run(go())


def test_cli_run_and_variants(tmp_path, capsys):
    from space.cli import main

    fasta = tmp_path / "in.fasta"
    fasta.write_text(FASTA)
    out = tmp_path / "run"
    main(["run", "-f", str(fasta), "--reference", REF, "-o", str(out), "--quiet", "--no-tree"])
    summary = json.loads(capsys.readouterr().out.split("\nTop")[0])
    assert summary["distinct_variants"] == 2
    main(["variants", str(out), "--format", "json", "--min-frequency", "0.3"])
    rows = json.loads(capsys.readouterr().out)
    assert [r["Mutation"] for r in rows] == ["I6G"]


def test_mcp_errors_are_readable(runs_dir):
    from mcp import Client

    from space.mcp_server import mcp

    async def go():
        async with Client(mcp) as client:
            res = await client.call_tool("get_variants", {"run_id": "no-such-run"})
            assert res.is_error and "Unknown run_id" in res.content[0].text

    asyncio.run(go())


def test_mcp_msa_resource_and_empty_run(runs_dir):
    from mcp import Client

    from space.mcp_server import mcp

    same = "\n".join(f">s{i}\n{REF}" for i in range(3))

    async def go():
        async with Client(mcp) as client:
            res = _payload(await client.call_tool("analyze_fasta", {"fasta": FASTA, "reference_sequence": REF}))
            msa = await client.read_resource(f"space://runs/{res['run_id']}/msa")
            assert msa.contents[0].text.startswith(">")
            empty = _payload(await client.call_tool("analyze_fasta", {"fasta": same, "reference_sequence": REF}))
            assert empty["distinct_variants"] == 0
            v = _payload(await client.call_tool("get_variants", {"run_id": empty["run_id"]}))
            assert v["variants"] == []

    asyncio.run(go())

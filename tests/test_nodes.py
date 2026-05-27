from __future__ import annotations

import cmg
from cmg.nodes import mint_id


def test_mint_id_uses_full_uuid_hex_suffix() -> None:
    node_id = mint_id("support")
    assert node_id.startswith("s-")
    assert len(node_id.removeprefix("s-")) == 32


def test_public_version_is_exported() -> None:
    assert isinstance(cmg.__version__, str)
    assert cmg.__version__

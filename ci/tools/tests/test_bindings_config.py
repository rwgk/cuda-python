# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import io
import json
from pathlib import Path

import pytest

from ci.tools.bindings_config import (
    BindingsConfigError,
    load_config,
    main,
    package_from_dict,
    validate_config,
)


def package(toolkit_version, release_status):
    return {
        "toolkit_version": toolkit_version,
        "release_status": release_status,
    }


def valid_config():
    return {
        "schema_version": 2,
        "cuda": {
            "bindings": {
                "package_roots": {
                    "cuda_bindings_12": package("12.9.1", "maintenance"),
                    "cuda_bindings": package("13.3.0", "current"),
                },
            }
        },
    }


def write_scm_config(root: Path, package_root: str, tag_regex: str) -> None:
    path = root / package_root / "pyproject.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"[tool.setuptools_scm]\ntag_regex = '{tag_regex}'\n", encoding="utf-8")


@pytest.mark.agent_authored(model="gpt-5.6")
def test_live_registry_has_ordered_package_roots_and_release_statuses():
    config = load_config()

    assert [package.package_root for package in config.package_roots] == ["cuda_bindings_12", "cuda_bindings"]
    assert config.package_for_release_status("current").package_root == "cuda_bindings"
    assert config.package_for_release_status("maintenance").package_root == "cuda_bindings_12"
    assert config.get_package("cuda_bindings_12").ctk_target == "12.9"
    assert config.get_package("cuda_bindings_12").tag_regex.startswith("^(?P<version>v12")
    assert config.get_package("cuda_bindings_12").cuda_major == "12"
    assert config.get_package("cuda_bindings_12").cuda_variant == "cu12"
    normalized = config.to_dict()
    assert [package["release_status"] for package in normalized["package_roots"]] == [
        "maintenance",
        "current",
    ]
    assert json.loads(config.to_json()) == normalized


@pytest.mark.agent_authored(model="gpt-5.6")
def test_tag_matching_uses_each_packages_scm_regex():
    config = validate_config(valid_config())
    assert config.match_tag("v12.9.8").package_root == "cuda_bindings_12"
    assert config.match_tag("v12.9.8a1") is None
    assert config.match_tag("v13.3.0b1").package_root == "cuda_bindings"
    assert config.match_tag("v13.3.0rc1").package_root == "cuda_bindings"
    assert config.match_tag("v13.3.0.dev1").package_root == "cuda_bindings"
    assert config.match_tag("v13.3.2.post1").package_root == "cuda_bindings"


@pytest.mark.agent_authored(model="gpt-5.6")
def test_public_registry_requires_distinct_cuda_abi_majors(tmp_path):
    raw = valid_config()
    bindings = raw["cuda"]["bindings"]
    bindings["package_roots"] = {
        "cuda_bindings_11_7": package("11.7.1", "maintenance"),
        "cuda_bindings_11_8": package("11.8.0", "current"),
    }
    write_scm_config(tmp_path, "cuda_bindings_11_7", r"^(?P<version>v11\.7\.\d+)$")
    write_scm_config(tmp_path, "cuda_bindings_11_8", r"^(?P<version>v11\.8\.\d+)$")

    with pytest.raises(BindingsConfigError, match="cuda_major values must be unique"):
        validate_config(raw, tmp_path)


@pytest.mark.parametrize(
    ("package_root", "message"),
    [
        ("../cuda_bindings_12", "normalized repository-relative POSIX path"),
        ("cuda_bindings_12\nINJECTED=value", "package root has invalid format"),
    ],
)
@pytest.mark.agent_authored(model="gpt-5.6")
def test_invalid_package_root_is_rejected(package_root, message):
    data = copy.deepcopy(valid_config())
    packages = data["cuda"]["bindings"]["package_roots"]
    maintenance = packages.pop("cuda_bindings_12")
    packages[package_root] = maintenance
    with pytest.raises(BindingsConfigError, match=message):
        validate_config(data)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_invalid_toolkit_version_is_rejected():
    data = copy.deepcopy(valid_config())
    data["cuda"]["bindings"]["package_roots"]["cuda_bindings_12"]["toolkit_version"] = "12.9"

    with pytest.raises(BindingsConfigError, match="toolkit_version has invalid format"):
        validate_config(data)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_load_wraps_yaml_errors(tmp_path):
    path = tmp_path / "versions.yml"
    path.write_text("cuda: [unterminated", encoding="utf-8")

    with pytest.raises(BindingsConfigError, match="could not read"):
        load_config(path)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_cli_emits_full_registry_and_selected_package(capsys):
    assert main([]) == 0
    registry = json.loads(capsys.readouterr().out)
    assert [package["package_root"] for package in registry["package_roots"]] == [
        "cuda_bindings_12",
        "cuda_bindings",
    ]

    assert main(["--package-roots"]) == 0
    packages = json.loads(capsys.readouterr().out)
    assert [package["package_root"] for package in packages] == ["cuda_bindings_12", "cuda_bindings"]

    assert main(["--release-status", "current"]) == 0
    current = json.loads(capsys.readouterr().out)
    assert current["package_root"] == "cuda_bindings"
    assert current["cuda_variant"] == "cu13"


@pytest.mark.agent_authored(model="gpt-5.6")
def test_cli_writes_package_json_from_stdin_to_github_env(tmp_path, capsys, monkeypatch):
    output = tmp_path / "github-env"
    package_json = load_config().package_for_release_status("current").to_dict()
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(package_json)))

    assert main(["write-github-env", str(output)]) == 0

    assert capsys.readouterr().out == ""
    assert output.read_text(encoding="utf-8").splitlines() == [
        "BUILD_CTK_VER=13.3.0",
        "BINDINGS_PACKAGE_ROOT=cuda_bindings",
        "BINDINGS_REGISTRY_ORIGIN=tag",
    ]


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_cli_write_github_env_requires_json_object(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("[]"))

    with pytest.raises(SystemExit, match="2"):
        main(["write-github-env", str(tmp_path / "github-env")])

    assert "stdin for write-github-env must contain a JSON object" in capsys.readouterr().err


@pytest.mark.agent_authored(model="gpt-5.6")
def test_list_valued_release_status_is_rejected():
    data = valid_config()
    data["cuda"]["bindings"]["package_roots"]["cuda_bindings_12"]["release_status"] = ["maintenance"]

    with pytest.raises(BindingsConfigError, match="release_status must be a non-empty, trimmed string"):
        validate_config(data)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_normalized_package_rejects_unknown_release_status():
    config = load_config()
    normalized = config.package_for_release_status("current").to_dict()
    normalized["release_status"] = "unknown"

    with pytest.raises(BindingsConfigError, match="must be one of current, maintenance"):
        package_from_dict(normalized)


@pytest.mark.parametrize(
    ("release_statuses", "message"),
    (
        (("current",), "exactly one current and one maintenance"),
        (("current", "current"), "exactly one current and one maintenance"),
    ),
)
@pytest.mark.agent_authored(model="gpt-5.6")
def test_public_release_statuses_must_cover_two_packages_once(release_statuses, message):
    data = valid_config()
    roots_and_versions = (("cuda_bindings_12", "12.9.1"), ("cuda_bindings", "13.3.0"))
    data["cuda"]["bindings"]["package_roots"] = {
        package_root: package(toolkit_version, release_status)
        for (package_root, toolkit_version), release_status in zip(
            roots_and_versions[: len(release_statuses)],
            release_statuses,
            strict=True,
        )
    }

    with pytest.raises(BindingsConfigError, match=message):
        validate_config(data)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_scm_regex_must_match_the_packages_toolkit_release(tmp_path):
    data = valid_config()
    write_scm_config(tmp_path, "cuda_bindings_12", r"^(?P<version>v13\.\d+\.\d+)$")
    write_scm_config(tmp_path, "cuda_bindings", r"^(?P<version>v13\.\d+\.\d+)$")

    with pytest.raises(BindingsConfigError, match="must match its configured toolkit release tag"):
        validate_config(data, tmp_path)

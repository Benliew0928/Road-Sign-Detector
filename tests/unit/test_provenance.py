from roadsign_assist.datasets.provenance import ProvenanceRegistry
from roadsign_assist.paths import project_path


def test_dataset_source_registry_is_valid() -> None:
    path = project_path("data/manifests/dataset_sources.json")
    registry = ProvenanceRegistry.model_validate_json(path.read_text(encoding="utf-8"))
    assert registry.sources
    assert all(source.decision != "rejected" for source in registry.sources)
    accepted = [source for source in registry.sources if source.decision == "accepted"]
    assert accepted
    assert all(source.licence_url for source in accepted)
    assert all(source.local_metadata_files for source in accepted)

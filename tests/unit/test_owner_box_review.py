import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


def module():
    path = Path(__file__).parents[2] / "scripts/owner_box_review.py"
    assert path.exists(), "Owner box review server has not been implemented"
    spec = importlib.util.spec_from_file_location("owner_box_review", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def store(tmp_path):
    mod = module()
    image = tmp_path / "source.png"
    Image.new("RGB", (200, 100), "white").save(image)
    row = {
        "inventory_id": "P7A-test",
        "image_path": str(image),
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "width": 200,
        "height": 100,
        "licence_status": "assignment_only",
        "use_policy": "no_redistribution",
        "initial_boxes": [],
        "owner_number": 20,
    }
    return mod.ReviewStore([row], tmp_path / "reviews"), row


def payload(**changes):
    return {
        "revision": 0,
        "status": "reviewed",
        "all_signs_checked": True,
        "notes": "",
        "boxes": [
            {"xyxy": [10, 20, 80, 90], "occluded": False, "truncated": False, "uncertain": False}
        ],
        **changes,
    }


def test_roundtrip_preserves_source_policy_and_history(store):
    s, row = store
    first = s.save(row["inventory_id"], payload())
    assert first["revision"] == 1
    assert first["training_allowed"] is False
    assert first["use_policy"] == "no_redistribution"
    assert first["semantic_labels_assigned"] is False
    assert s.get(row["inventory_id"])["review"] == first
    s.save(row["inventory_id"], payload(revision=1, status="draft"))
    assert len(list((s.output / "history").glob("*.json"))) == 2
    assert hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest() == row["image_sha256"]


@pytest.mark.parametrize(
    "box", [[0, 0, 201, 90], [50, 10, 20, 90], [0, 0, float("nan"), 90], [0, 0, 0, 0]]
)
def test_invalid_geometry_cannot_be_saved(store, box):
    s, row = store
    with pytest.raises(ValueError):
        s.save(row["inventory_id"], payload(boxes=[{"xyxy": box}]))


def test_uncertainty_cannot_be_marked_complete(store):
    s, row = store
    p = payload()
    p["boxes"][0]["uncertain"] = True
    with pytest.raises(ValueError, match="uncertain"):
        s.save(row["inventory_id"], p)
    p["status"] = "needs_review"
    assert s.save(row["inventory_id"], p)["complete_scene_review"] is False


def test_empty_and_unchecked_reviews_are_not_negatives(store):
    s, row = store
    for p in [
        payload(boxes=[]),
        payload(all_signs_checked=False),
        payload(status="no_sign", boxes=[], all_signs_checked=False),
        payload(status="no_sign"),
    ]:
        with pytest.raises(ValueError):
            s.save(row["inventory_id"], p)
    r = s.save(row["inventory_id"], payload(status="no_sign", boxes=[]))
    assert r["training_allowed"] is False
    assert r["status"] == "no_sign"


def test_stale_save_and_changed_source_are_rejected(store):
    s, row = store
    s.save(row["inventory_id"], payload())
    with pytest.raises(ValueError, match="revision"):
        s.save(row["inventory_id"], payload())
    Path(row["image_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        s.save(row["inventory_id"], payload(revision=1))


def test_ids_cannot_escape_review_directory(store):
    s, _ = store
    with pytest.raises(KeyError):
        s.get("../secret")
    with pytest.raises(KeyError):
        s.save("../secret", payload())


def test_assistant_proposal_is_visible_but_not_counted_as_owner_review(tmp_path):
    mod = module()
    image = tmp_path / "source.png"
    Image.new("RGB", (200, 100), "white").save(image)
    ident = "P7A-proposal"
    row = {
        "inventory_id": ident,
        "image_path": str(image),
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "width": 200,
        "height": 100,
        "licence_status": "assignment_only",
        "use_policy": "no_redistribution",
        "initial_boxes": [],
    }
    proposals = tmp_path / "proposals"
    proposals.mkdir()
    proposal = {"status": "assistant_proposal", "boxes": [{"xyxy": [1, 2, 30, 40]}]}
    (proposals / f"{ident}.json").write_text(json.dumps(proposal))
    store = mod.ReviewStore([row], tmp_path / "reviews", proposals)
    assert store.get(ident)["review"] is None
    assert store.get(ident)["assistant_proposal"] == proposal
    assert store.index()[0]["status"] == "assistant_proposal"
    assert not (tmp_path / "reviews" / f"{ident}.json").exists()


def test_manifest_pins_inputs_and_resume_keeps_queue_order(tmp_path):
    mod = module()
    p = tmp_path / "package"
    (p / "annotation_assist_v1").mkdir(parents=True)
    (p / "visual_audit_v1").mkdir()
    image = tmp_path / "source.png"
    Image.new("RGB", (20, 10)).save(image)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    rows = []
    for ident in ["P7A-a", "P7A-b"]:
        r = {
            "inventory_id": ident,
            "image_path": str(image),
            "image_sha256": digest,
            "width": 20,
            "height": 10,
            "inherited_boxes": [[0, 0, 10, 10]],
        }
        (p / "annotation_assist_v1" / f"{ident}.json").write_text(json.dumps(r))
        rows.append(
            {**r, "sha256": digest, "licence_status": "restricted", "use_policy": "restricted"}
        )
    (p / "visual_audit_v1/PRIORITIZED_DETECTOR_REVIEW.json").write_text(json.dumps(rows))
    (p / "DETAIL_REVIEW_QUEUE.json").write_text(
        json.dumps({"remaining_uninspected_ids": ["P7A-a", "P7A-b"], "inspected_hold_ids": []})
    )
    owner = p / "optional_owner_review_v6"
    owner.mkdir()
    (owner / "REVIEW_ASSIGNMENT.json").write_text(
        json.dumps({"rows": [{"inventory_id": "P7A-b", "number": 20}]})
    )
    manifest = mod.prepare_manifest(p)
    assert [r["inventory_id"] for r in manifest["rows"]] == ["P7A-b", "P7A-a"]
    (p / "DETAIL_REVIEW_QUEUE.json").write_text("{}")
    assert mod.prepare_manifest(p) == manifest


def test_http_save_origin_guards_and_image_pixels(store):
    import io
    import threading
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    s, row = store
    server = module().make_server(s, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/index") as response:
            token = json.load(response)["token"]
        with urlopen(base + "/api/image/" + row["inventory_id"]) as response:
            im = Image.open(io.BytesIO(response.read()))
            assert im.size == (200, 100)
            assert im.getpixel((50, 50)) == (255, 255, 255)
        endpoint = base + "/api/review/" + row["inventory_id"]
        body = json.dumps(payload()).encode()
        with pytest.raises(HTTPError) as error:
            urlopen(Request(endpoint, data=body))
        assert error.value.code == 403
        headers = {"Origin": base, "X-Review-Token": token, "Content-Type": "application/json"}
        with urlopen(Request(endpoint, data=body, headers=headers)) as response:
            assert json.load(response)["revision"] == 1
        with urlopen(base + "/api/export") as response:
            exported = json.load(response)
            assert len(exported["reviews"]) == 1
            assert exported["training_allowed"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

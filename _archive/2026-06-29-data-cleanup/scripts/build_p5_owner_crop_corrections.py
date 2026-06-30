from __future__ import annotations

import csv
from pathlib import Path


MANIFEST_PATH = Path("data/manifests/p5_label_qc_manifest.csv")
OUTPUT_PATH = Path("data/manifests/p5_owner_crop_corrections.csv")


OWNER_CORRECTIONS: dict[str, tuple[str, str]] = {
    # Source-level remap exceptions found during owner P5 review.
    "emtd_27fc894ec54867c0_001": ("give_way", "source class 15 exception: visible give-way sign"),
    "emtd_4486c2d99d3f2790_002": ("no_stopping", "source class 15 exception: visible no-stopping sign"),
    "emtd_5c18f71806f7f407_002": ("height_restriction", "source class 24 exception: visible 5.4 m height limit"),
    "emtd_c6ff2feb98444e60_002": ("height_restriction", "source class 24 exception: visible 5.4 m height limit"),
    "emtd_83c81868dff163c0_002": ("maximum_speed", "source class 32 exception: visible speed-limit sign"),

    # Current keep/side-road and old merge-folder exceptions.
    "emtd_69a204a515f07cc4_001": ("side_road_right", "owner review: keep_left crop is side road right"),
    "emtd_3ed7a222fd38cc5f_002": ("side_road_right", "owner review: keep_left crop is side road right"),
    "emtd_9f4c563b497eb46b_012": ("permitted_u_turn", "owner review: old merge_left crop is permitted U-turn"),
    "emtd_9f4c563b497eb46b_013": ("height_restriction", "owner review: old merge_left crop is height restriction"),
    "emtd_f46f86b31a823c41_011": ("obstruction_ahead", "owner review: old merge_left crop is obstruction marker"),
    "emtd_f46f86b31a823c41_012": ("obstruction_ahead", "owner review: old merge_left crop is obstruction marker"),
    "emtd_69a204a515f07cc4_012": ("give_way", "owner review: old merge_left crop is give way"),
    "emtd_69a204a515f07cc4_011": ("pedestrian_crossing", "owner review: old merge_left crop is pedestrian crossing"),
    "emtd_9f4c563b497eb46b_011": ("give_way", "owner review: old merge_right crop is give way"),
    "emtd_69a204a515f07cc4_010": ("no_u_turn", "owner review: old merge_right crop is no U-turn"),
    "emtd_3ed7a222fd38cc5f_013": ("height_restriction", "owner review: old merge_right crop is height restriction"),
    "emtd_f46f86b31a823c41_010": ("obstruction_ahead", "owner visual check: side_road_right crop is obstruction ahead"),

    # Divided-road-begins exceptions.
    "emtd_9f4c563b497eb46b_014": ("obstruction_ahead", "owner review: divided_road_begins crop is obstruction ahead"),
    "emtd_872688bce44ccd84_002": ("obstruction_ahead", "owner review: divided_road_begins crop is obstruction ahead"),
    "emtd_f46f86b31a823c41_013": ("no_entry", "owner review: divided_road_begins crop is no entry"),
    "emtd_3ed7a222fd38cc5f_015": ("give_way", "owner review: divided_road_begins crop is give way"),

    # Maximum-speed exceptions.
    "emtd_803a3df237414bc0_002": ("no_u_turn", "owner review: maximum_speed crop is no U-turn"),
    "emtd_009e580a009c7377_001": ("no_parking", "owner review: maximum_speed crop is no parking"),
    "emtd_8b84acee490ad60d_002": ("height_restriction", "owner review: maximum_speed crop is height restriction"),
    "emtd_5f7cf79a455c49f0_004": ("height_restriction", "owner review: maximum_speed crop is height restriction"),
    "emtd_05c11e440bceae36_002": ("height_restriction", "owner review: maximum_speed crop is height restriction"),
    "emtd_83c81868dff163c0_001": ("height_restriction", "owner review: maximum_speed crop is height restriction"),
    "emtd_04128cdf51be3987_002": ("height_restriction", "owner review: maximum_speed crop is height restriction"),
    "emtd_3ed7a222fd38cc5f_007": ("general_caution", "owner review: maximum_speed crop is general caution"),
    "emtd_e88a6cbfc3b37e6b_002": ("weight_restriction", "owner review: maximum_speed crop is weight restriction"),

    # No-entry exceptions.
    "emtd_9f4c563b497eb46b_004": ("general_caution", "owner review: no_entry crop is general caution"),
    "emtd_f46f86b31a823c41_004": ("side_road_right", "owner review: no_entry crop is side road right"),
    "emtd_4dafc2a4b3a3316b_003": ("give_way", "owner review: no_entry crop is give way"),
    "emtd_4dafc2a4b3a3316b_004": ("no_u_turn", "owner review: no_entry crop is no U-turn"),
    "emtd_3ed7a222fd38cc5f_004": ("divided_road_begins", "owner follow-up review: obstruction_ahead crop is divided road begins"),
    "emtd_69a204a515f07cc4_003": ("side_road_left", "owner review: no_entry crop is side road left"),
    "emtd_69a204a515f07cc4_004": ("side_road_left", "owner review: no_entry crop is side road left"),
    "emtd_4979f8f85583d8ec_002": ("no_straight_ahead", "owner visual check: no_entry crop has vertical prohibition bar"),
    "emtd_4979f8f85583d8ec_003": ("no_straight_ahead", "owner visual check: no_entry crop has vertical prohibition bar"),
    "emtd_ae3c2d8d4035e435_001": ("no_straight_ahead", "owner visual check: no_entry crop has vertical prohibition bar"),
    "emtd_ae3c2d8d4035e435_002": ("no_straight_ahead", "owner visual check: no_entry crop has vertical prohibition bar"),
    "emtd_ae3c2d8d4035e435_003": ("no_straight_ahead", "owner visual check: no_entry crop has vertical prohibition bar"),
    "emtd_ae3c2d8d4035e435_004": ("no_straight_ahead", "owner visual check: no_entry crop has vertical prohibition bar"),
    "emtd_ae3c2d8d4035e435_005": ("no_straight_ahead", "owner visual check: no_entry crop has vertical prohibition bar"),

    # No-parking and no-stopping exceptions.
    "emtd_009e580a009c7377_003": ("maximum_speed", "owner review: no_parking crop is speed limit 50"),
    "emtd_0c571b3b7d5bcf8d_003": ("height_restriction", "owner review: no_parking crop is height restriction"),
    "emtd_f032064e939cabdc_001": ("give_way", "owner review: no_parking crop is give way"),
    "emtd_f4c05b290ec60cf7_002": ("no_stopping", "owner review: no_parking crop is no stopping"),
    "emtd_4a175710f8d593a0_001": ("no_stopping", "owner review: no_parking crop is no stopping"),
    "emtd_4a175710f8d593a0_002": ("no_parking", "owner follow-up review: no_stopping crop is no parking"),

    # No-U-turn exceptions.
    "emtd_803a3df237414bc0_001": ("maximum_speed", "owner review: no_u_turn crop is maximum speed"),
    "emtd_9f4c563b497eb46b_003": ("side_road_left", "owner review: no_u_turn crop is side road left"),
    "emtd_f46f86b31a823c41_003": ("side_road_right", "owner review: no_u_turn crop is side road right"),
    "emtd_3888248c925e9c81_001": ("no_stopping", "owner review: no_u_turn crop is no stopping"),
    "emtd_4dafc2a4b3a3316b_002": ("no_entry", "owner review: no_u_turn crop is no entry"),
    "emtd_38cf0bbf0e297237_002": ("give_way", "owner review: no_u_turn crop is give way"),
    "emtd_3ed7a222fd38cc5f_003": ("obstruction_ahead", "owner review: no_u_turn crop is obstruction marker"),
    "emtd_69a204a515f07cc4_002": ("obstruction_ahead", "owner review: no_u_turn crop is obstruction marker"),

    # Other owner-reviewed crop exceptions.
    "emtd_4486c2d99d3f2790_001": ("no_stopping", "owner review: pass_either_side crop is no stopping"),
    "emtd_9f4c563b497eb46b_007": ("no_u_turn", "owner review: pedestrian_crossing crop is no U-turn"),
    "emtd_f46f86b31a823c41_007": ("permitted_u_turn", "owner review: pedestrian_crossing crop is permitted U-turn"),
    "emtd_3ed7a222fd38cc5f_009": ("no_entry", "owner review: pedestrian_crossing crop is no entry"),
    "emtd_69a204a515f07cc4_007": ("no_entry", "owner review: pedestrian_crossing crop is no entry"),
    "emtd_ce9c16ca0280d757_002": ("height_restriction", "owner review: stop crop is height restriction"),
    "emtd_e88a6cbfc3b37e6b_003": ("maximum_speed", "owner review: weight_restriction crop is speed limit 50"),

    # General-caution and give-way cleanups after source-level remaps.
    "emtd_3ed7a222fd38cc5f_010": ("permitted_u_turn", "owner visual check: general_caution crop is permitted U-turn"),
    "emtd_9f4c563b497eb46b_008": ("side_road_left", "owner visual check: general_caution crop is side road left"),
    "emtd_9f4c563b497eb46b_006": ("side_road_left", "owner review: give_way crop is side road left"),
    "emtd_f46f86b31a823c41_006": ("side_road_right", "owner review: give_way crop is side road right"),
    "emtd_27fc894ec54867c0_004": ("no_stopping", "owner visual check: give_way crop is no stopping"),
    "emtd_38cf0bbf0e297237_003": ("no_u_turn", "owner visual check: give_way crop is no U-turn"),
    "emtd_4dafc2a4b3a3316b_005": ("no_entry", "owner visual check: give_way crop is no entry"),
    "emtd_7b1961bec90c74cc_002": ("no_left_turn", "owner visual check: give_way crop is no left turn"),
    "emtd_ecf89bd784bba2e2_002": ("no_stopping", "owner visual check: give_way crop is no stopping"),
    "emtd_f032064e939cabdc_003": ("no_parking", "owner visual check: give_way crop is no parking"),
    "emtd_3ed7a222fd38cc5f_008": ("pedestrian_crossing", "owner follow-up review: obstruction_ahead crop is pedestrian crossing"),
    "emtd_69a204a515f07cc4_006": ("keep_left", "owner visual check: give_way crop is keep-left arrow"),

    # Owner follow-up review on 2026-06-27: maximum speed and height restriction cleanup.
    "emtd_595647f10f1a4e3e_003": ("height_restriction", "owner follow-up review: maximum_speed crop is height restriction"),
    "emtd_2b0800c50aaa1f8c_002": ("height_restriction", "owner follow-up review: maximum_speed crop is height restriction"),
    "emtd_4486c2d99d3f2790_010": ("no_stopping", "owner follow-up review: vehicle_collision_hazard crop is no stopping"),
    "emtd_3ed7a222fd38cc5f_005": ("obstruction_ahead", "owner follow-up review: height_restriction crop is obstruction ahead"),
    "emtd_3ed7a222fd38cc5f_006": ("keep_left", "owner follow-up review: height_restriction crop is keep left"),
    "emtd_4a175710f8d593a0_003": ("no_parking", "owner follow-up review: height_restriction crop is no parking"),
    "emtd_0c571b3b7d5bcf8d_001": ("no_parking", "owner follow-up review: height_restriction crop is no parking"),
    "emtd_c6ff2feb98444e60_001": ("maximum_speed", "owner follow-up review: height_restriction crop is maximum speed"),
    "emtd_05c11e440bceae36_001": ("maximum_speed", "owner follow-up review: height_restriction crop is maximum speed"),
    "emtd_5c18f71806f7f407_001": ("maximum_speed", "owner follow-up review: height_restriction crop is maximum speed"),
    "emtd_5f7cf79a455c49f0_002": ("maximum_speed", "owner follow-up review: height_restriction crop is maximum speed"),
    "emtd_2b0800c50aaa1f8c_001": ("maximum_speed", "owner follow-up review: height_restriction crop is maximum speed"),
    "emtd_04128cdf51be3987_001": ("maximum_speed", "owner follow-up review: height_restriction crop is maximum speed"),
    "emtd_5975b4f591e8f67c_001": ("maximum_speed", "owner follow-up review: height_restriction crop is maximum speed"),
    "emtd_595647f10f1a4e3e_002": ("maximum_speed", "owner follow-up review: height_restriction crop is maximum speed"),
    "emtd_8b84acee490ad60d_001": ("maximum_speed", "owner follow-up review: height_restriction crop is maximum speed"),
    "emtd_ff937d8340aa908c_001": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_6b3bad116f3adbf1_002": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_724032002f9a00a9_001": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_f032064e939cabdc_002": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_f43403930d9e849f_001": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_f8082107503b594d_002": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_fe3876c03658671d_002": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_e922be64f6c9a29b_001": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_eed9757fa6d6f8ad_001": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_6f9e795a68442be9_001": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_27fc894ec54867c0_003": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_28e4e221ad933605_003": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_96c840a754c9d933_001": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_680f6d595734dbcd_001": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_744806c4a943abab_002": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_51965458ff11aa25_001": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_110456139d0a1273_002": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_695985373825108d_001": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_a2bb22296a921e05_001": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_0a80e5fcdaaf3765_003": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_803a3df237414bc0_003": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_4486c2d99d3f2790_004": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_4486c2d99d3f2790_005": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_4832e5835a0f1e95_001": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_214262c5339b209c_002": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_ce9c16ca0280d757_003": ("stop", "owner follow-up review: height_restriction crop is stop sign"),
    "emtd_ecf89bd784bba2e2_001": ("give_way", "owner follow-up review: height_restriction crop is give way"),
    "emtd_9f4c563b497eb46b_005": ("divided_road_begins", "owner follow-up review: height_restriction crop is divided road begins"),
    "emtd_4486c2d99d3f2790_007": ("pass_either_side", "owner follow-up review: height_restriction crop is pass either side"),
    "emtd_4486c2d99d3f2790_006": ("vehicle_collision_hazard", "owner follow-up review: height_restriction crop is vehicle collision hazard"),
    "emtd_c2a0f4313c3c9e12_001": ("no_stopping", "owner follow-up review: height_restriction crop is no stopping"),
    "emtd_f4c05b290ec60cf7_003": ("no_parking", "owner follow-up review: height_restriction crop is no parking"),
    "emtd_f46f86b31a823c41_005": ("side_road_left", "owner follow-up review: height_restriction crop is side road left"),
}


OWNER_CORRECTIONS.update(
    {
        # Owner follow-up review on 2026-06-27: split merge signs out of side-road labels.
        "emtd_1f2671d8396a5778_003": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_3ed7a222fd38cc5f_014": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_69a204a515f07cc4_003": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_69a204a515f07cc4_004": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_5959063e85e67c7b_003": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_bf46fd2a09e05bed_007": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_1ec87425fce92b2f_001": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_3b6a8f1eafa9f748_001": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_4dafc2a4b3a3316b_009": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_5df46409ee99afa6_007": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_5fce6e09bed7a483_003": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_5fce6e09bed7a483_004": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_5fce6e09bed7a483_005": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_030c8af46287be06_002": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_30d3540ab66976b8_001": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_38d2092e8a2e7e09_001": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_8673f35e4b6971c2_002": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_20923a7ba318eb58_004": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_606776c64a392220_002": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_ade5e4cb73268bc1_001": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_ccbea56843be030f_002": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_d235b581fa497e34_003": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_df679b3a4a2692a0_002": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_e88f80e2db86cbb7_003": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_e968b703a29148d9_003": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_f9af889d5c2a88d0_006": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_4e0edc87f7466c6a_007": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_9f4c563b497eb46b_003": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_9f4c563b497eb46b_006": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_9f4c563b497eb46b_008": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_65e9002f35db15b4_001": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_b330bbc32b124384_003": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_ed6369453cc5fd41_006": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_f46f86b31a823c41_005": ("merge_left", "owner follow-up review: side_road_left crop is merge-left sign"),
        "emtd_3ed7a222fd38cc5f_002": ("merge_right", "owner follow-up review: side_road_right crop is merge-right sign"),
        "emtd_69a204a515f07cc4_001": ("merge_right", "owner follow-up review: side_road_right crop is merge-right sign"),
        "emtd_fe3876c03658671d_004": ("merge_right", "owner follow-up review: side_road_right crop is merge-right sign"),
        "emtd_1db3d5c40c201627_002": ("merge_right", "owner follow-up review: side_road_right crop is merge-right sign"),
        "emtd_4dafc2a4b3a3316b_008": ("merge_right", "owner follow-up review: side_road_right crop is merge-right sign"),
        "emtd_6c6336a4fe5b9a86_002": ("merge_right", "owner follow-up review: side_road_right crop is merge-right sign"),
        "emtd_9aa7dca4b8a09a1d_002": ("merge_right", "owner follow-up review: side_road_right crop is merge-right sign"),
        "emtd_86ce16e7b74daf8f_008": ("merge_right", "owner follow-up review: side_road_right crop is merge-right sign"),
        "emtd_90b477b11d9c37f6_001": ("merge_right", "owner follow-up review: side_road_right crop is merge-right sign"),
        "emtd_97ad7887eb642fb9_006": ("merge_right", "owner follow-up review: side_road_right crop is merge-right sign"),
        "emtd_d81b8450e1f6b24c_006": ("merge_right", "owner follow-up review: side_road_right crop is merge-right sign"),
        "emtd_f46f86b31a823c41_003": ("merge_right", "owner follow-up review: side_road_right crop is merge-right sign"),
        "emtd_f46f86b31a823c41_006": ("merge_right", "owner follow-up review: side_road_right crop is merge-right sign"),
        "emtd_f46f86b31a823c41_004": ("pedestrian_crossing", "owner follow-up review: side_road_right crop is pedestrian crossing"),
    }
)


FIELDNAMES = [
    "action",
    "status",
    "split",
    "current_label",
    "new_label",
    "file",
    "instance_id",
    "source_class_id",
    "reason",
    "reviewer",
]


def main() -> None:
    with MANIFEST_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["instance_id"]: row for row in csv.DictReader(handle)}

    corrections: list[dict[str, str]] = []
    missing: list[str] = []
    already_correct: list[str] = []
    for instance_id, (new_label, reason) in sorted(OWNER_CORRECTIONS.items()):
        row = rows.get(instance_id)
        if row is None:
            missing.append(instance_id)
            continue
        status = "approved"
        if row["current_label"] == new_label:
            already_correct.append(instance_id)
            status = "already_correct"
        corrections.append(
            {
                "action": "relabel",
                "status": status,
                "split": row["split"],
                "current_label": row["current_label"],
                "new_label": new_label,
                "file": row["file"],
                "instance_id": instance_id,
                "source_class_id": row["source_class_id"],
                "reason": reason,
                "reviewer": "owner",
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(corrections)

    pending_count = sum(row["status"] == "approved" for row in corrections)
    print(f"Wrote {len(corrections)} owner crop decisions to {OUTPUT_PATH}")
    print(f"{pending_count} corrections still need applying.")
    if already_correct:
        print(f"Skipped {len(already_correct)} already-correct crops.")
    if missing:
        raise SystemExit(f"Missing instance IDs in current P5 manifest: {', '.join(missing)}")


if __name__ == "__main__":
    main()

# ruff: noqa: RUF001

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from roadsign_assist.catalogue.models import ActionCode, Severity, SignDefinition
from roadsign_assist.catalogue.repository import load_catalogue

LanguageCode = Literal["en", "ms", "zh"]

LANGUAGES: tuple[LanguageCode, ...] = ("en", "ms", "zh")
SPEED_VALUES_KMH = (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160)
HEIGHT_VALUES_M = (2.0, 2.5, 3.0, 3.5, 4.0, 4.2, 4.5, 5.0, 5.5)
WIDTH_VALUES_M = (2.0, 2.3, 2.5, 3.0, 3.5, 4.0, 4.5)
WEIGHT_VALUES_T = (3, 5, 7, 8, 10, 16, 20, 30)

PRIORITY_BY_SEVERITY: dict[str, int] = {
    Severity.INFORMATION.value: 1,
    Severity.CAUTION.value: 2,
    Severity.WARNING.value: 3,
    Severity.CRITICAL.value: 4,
}


@dataclass(frozen=True)
class PhraseText:
    en: str
    ms: str
    zh: str

    def model_dump(self) -> dict[str, str]:
        return {"en": self.en, "ms": self.ms, "zh": self.zh}


def _name(entry: SignDefinition, lang: LanguageCode) -> str:
    return getattr(entry.names, lang)


def _format_number(value: float | int) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).rstrip("0").rstrip(".")


def phrase_id_value(value: float | int) -> str:
    return _format_number(value).replace(".", "_")


def _speed_text(value: int, *, minimum: bool = False, temporary: bool = False) -> PhraseText:
    if minimum:
        return PhraseText(
            en=(
                f"The minimum speed here is {value} kilometers per hour. "
                "Keep moving safely and avoid driving too slowly for the traffic flow."
            ),
            ms=(
                f"Had laju minimum di sini ialah {value} kilometer sejam. "
                "Teruskan pemanduan dengan selamat dan jangan terlalu perlahan hingga mengganggu aliran trafik."
            ),
            zh=(
                f"这里的最低车速是每小时{value}公里。请保持安全车速，不要过慢影响交通流动。"
            ),
        )
    prefix_en = "This temporary road section" if temporary else "This road"
    prefix_ms = "Bahagian jalan sementara ini" if temporary else "Jalan ini"
    prefix_zh = "这段临时道路" if temporary else "这条路"
    return PhraseText(
        en=(
            f"{prefix_en} has a speed limit of {value} kilometers per hour. "
            "Please keep your speed below the limit and leave enough space ahead."
        ),
        ms=(
            f"{prefix_ms} mempunyai had laju {value} kilometer sejam. "
            "Sila pastikan kelajuan tidak melebihi had dan kekalkan jarak selamat."
        ),
        zh=f"{prefix_zh}的限速是每小时{value}公里。请不要超速，并与前车保持安全距离。",
    )


def _height_text(value: float) -> PhraseText:
    number = _format_number(value)
    return PhraseText(
        en=(
            f"Height is restricted to {number} meters ahead. "
            "Check your vehicle clearance before continuing under this route."
        ),
        ms=(
            f"Had tinggi di hadapan ialah {number} meter. "
            "Pastikan ketinggian kenderaan selamat sebelum meneruskan perjalanan."
        ),
        zh=f"前方限高{number}米。继续通过前，请确认车辆高度可以安全通过。",
    )


def _width_text(value: float) -> PhraseText:
    number = _format_number(value)
    return PhraseText(
        en=(
            f"Vehicle width is restricted to {number} meters ahead. "
            "Keep centered and do not enter if your vehicle is too wide."
        ),
        ms=(
            f"Had lebar kenderaan di hadapan ialah {number} meter. "
            "Pastikan kenderaan berada di tengah lorong dan jangan masuk jika terlalu lebar."
        ),
        zh=f"前方限宽{number}米。请保持在车道中央，车辆过宽时不要进入。",
    )


def _weight_text(value: int) -> PhraseText:
    return PhraseText(
        en=(
            f"Vehicle weight is restricted to {value} tonnes ahead. "
            "If your vehicle is heavier than this, choose another route."
        ),
        ms=(
            f"Had berat kenderaan di hadapan ialah {value} tan. "
            "Jika kenderaan anda lebih berat, sila gunakan laluan lain."
        ),
        zh=f"前方限重{value}吨。如果车辆超过这个重量，请选择其他路线。",
    )


def _direction_word(direction: str | None, lang: LanguageCode) -> str:
    words = {
        "left": {"en": "left", "ms": "kiri", "zh": "左"},
        "right": {"en": "right", "ms": "kanan", "zh": "右"},
        "straight": {"en": "straight ahead", "ms": "terus ke hadapan", "zh": "直行"},
        "u_turn": {"en": "make a U-turn", "ms": "membuat pusingan U", "zh": "掉头"},
        "straight_or_left": {"en": "straight or left", "ms": "terus atau kiri", "zh": "直行或左转"},
        "straight_or_right": {"en": "straight or right", "ms": "terus atau kanan", "zh": "直行或右转"},
        "left_or_right": {"en": "left or right", "ms": "kiri atau kanan", "zh": "左转或右转"},
        "either_side": {"en": "either side", "ms": "mana-mana sisi", "zh": "任一侧"},
        "roundabout": {"en": "around the roundabout", "ms": "mengikut bulatan", "zh": "按环岛方向"},
    }
    return words.get(direction or "", {}).get(lang, "")


def _generic_entry_text(entry: SignDefinition) -> PhraseText:
    action = entry.base_action
    direction = str(entry.default_parameter) if entry.default_parameter else None
    name_en = _name(entry, "en")
    name_ms = _name(entry, "ms")
    name_zh = _name(entry, "zh")

    if action is ActionCode.STOP_REQUEST:
        return PhraseText(
            en=f"{name_en} ahead. Please slow down early, stop completely, and continue only when it is safe.",
            ms=f"{name_ms} di hadapan. Sila perlahankan kenderaan, berhenti sepenuhnya, dan teruskan hanya apabila selamat.",
            zh=f"前方{name_zh}。请提前减速，完全停车，确认安全后再继续。",
        )
    if action is ActionCode.YIELD:
        return PhraseText(
            en=f"{name_en} ahead. Slow down and give priority to traffic that already has the right of way.",
            ms=f"{name_ms} di hadapan. Perlahankan kenderaan dan beri laluan kepada trafik yang mempunyai keutamaan.",
            zh=f"前方{name_zh}。请减速，并让有优先权的车辆先行。",
        )
    if action is ActionCode.PROHIBIT_ENTRY:
        return PhraseText(
            en="No entry ahead. Do not continue into this road; prepare to choose another safe route.",
            ms="Dilarang masuk di hadapan. Jangan teruskan ke jalan ini; bersedia untuk menggunakan laluan lain yang selamat.",
            zh="前方禁止驶入。请不要继续进入这条道路，并准备选择其他安全路线。",
        )
    if action in {ActionCode.PROHIBIT_LEFT_TURN, ActionCode.PROHIBIT_RIGHT_TURN, ActionCode.PROHIBIT_U_TURN, ActionCode.PROHIBIT_DIRECTION}:
        en_direction = _direction_word(direction, "en") or "that direction"
        ms_direction = _direction_word(direction, "ms") or "arah tersebut"
        zh_direction = _direction_word(direction, "zh") or "这个方向"
        return PhraseText(
            en=f"{name_en}. Do not go {en_direction}; keep following the permitted route safely.",
            ms=f"{name_ms}. Jangan bergerak ke arah {ms_direction}; teruskan pada laluan yang dibenarkan dengan selamat.",
            zh=f"{name_zh}。请不要往{zh_direction}行驶，继续沿允许的路线安全行驶。",
        )
    if action is ActionCode.PROHIBIT_LANE_CHANGE:
        return PhraseText(
            en="Lane changing is not allowed here. Stay in your lane and avoid sudden steering.",
            ms="Pertukaran lorong tidak dibenarkan di sini. Kekal di lorong anda dan elakkan stereng mengejut.",
            zh="这里禁止变换车道。请保持在当前车道，避免突然转向。",
        )
    if action is ActionCode.PROHIBIT_OVERTAKING:
        return PhraseText(
            en="Overtaking is not allowed here. Stay behind the vehicle ahead until the road permits passing.",
            ms="Memotong tidak dibenarkan di sini. Kekal di belakang kenderaan hadapan sehingga laluan membenarkan memotong.",
            zh="这里禁止超车。请跟随前车，等道路允许时再超越。",
        )
    if action is ActionCode.PROHIBIT_VEHICLE:
        return PhraseText(
            en=f"{name_en}. This vehicle type is not allowed here, so do not enter if it applies to you.",
            ms=f"{name_ms}. Jenis kenderaan ini tidak dibenarkan di sini, jadi jangan masuk jika berkaitan dengan kenderaan anda.",
            zh=f"{name_zh}。这种车辆不允许进入，请确认适用时不要驶入。",
        )
    if action is ActionCode.PROHIBIT_PARKING:
        return PhraseText(
            en="Parking is not allowed here. Keep moving and look for a proper parking area.",
            ms="Meletak kenderaan tidak dibenarkan di sini. Teruskan perjalanan dan cari kawasan parkir yang sesuai.",
            zh="这里禁止停车。请继续行驶，并寻找合适的停车地点。",
        )
    if action is ActionCode.PROHIBIT_STOPPING:
        return PhraseText(
            en="Stopping is not allowed here. Keep the vehicle moving unless there is an emergency.",
            ms="Berhenti tidak dibenarkan di sini. Teruskan pergerakan kecuali dalam kecemasan.",
            zh="这里禁止停车。除非紧急情况，请保持车辆继续行驶。",
        )
    if action is ActionCode.PROHIBIT_HORN:
        return PhraseText(
            en="Horn use is not allowed here. Drive quietly and stay alert to nearby road users.",
            ms="Membunyikan hon tidak dibenarkan di sini. Pandu dengan tenang dan peka terhadap pengguna jalan lain.",
            zh="这里禁止鸣笛。请安静驾驶，并注意周围道路使用者。",
        )
    if action in {ActionCode.KEEP_LEFT, ActionCode.KEEP_RIGHT, ActionCode.FOLLOW_DIRECTION}:
        en_direction = _direction_word(direction, "en") or name_en.lower()
        ms_direction = _direction_word(direction, "ms") or name_ms.lower()
        zh_direction = _direction_word(direction, "zh") or name_zh
        return PhraseText(
            en=f"{name_en}. Follow the required direction, {en_direction}, and signal early if you need to adjust position.",
            ms=f"{name_ms}. Ikut arah yang diwajibkan, {ms_direction}, dan beri isyarat awal jika perlu menukar posisi.",
            zh=f"{name_zh}。请按规定方向{zh_direction}行驶，如需调整位置请提前打灯。",
        )
    if action is ActionCode.SOUND_HORN:
        return PhraseText(
            en="Sound horn if needed ahead. Warn others gently and continue with care.",
            ms="Bunyikan hon jika perlu di hadapan. Beri amaran dengan berhemah dan teruskan dengan berhati-hati.",
            zh="前方需要时请鸣笛。请适度提醒他人，并谨慎通过。",
        )
    if action is ActionCode.WATCH_PEDESTRIANS:
        return PhraseText(
            en=f"{name_en} ahead. Slow down and be ready for people crossing or standing near the road.",
            ms=f"{name_ms} di hadapan. Perlahankan kenderaan dan bersedia untuk pejalan kaki yang melintas atau berada berhampiran jalan.",
            zh=f"前方{name_zh}。请减速，并注意可能过马路或在路边的行人。",
        )
    if action is ActionCode.WATCH_CHILDREN:
        return PhraseText(
            en=f"{name_en} ahead. Reduce speed and be extra careful because children may move unpredictably.",
            ms=f"{name_ms} di hadapan. Kurangkan kelajuan dan lebih berhati-hati kerana kanak-kanak mungkin bergerak secara tiba-tiba.",
            zh=f"前方{name_zh}。请降低车速，儿童可能突然移动，需要格外小心。",
        )
    if action is ActionCode.WATCH_CYCLISTS:
        return PhraseText(
            en=f"{name_en} ahead. Slow down and give cyclists enough space.",
            ms=f"{name_ms} di hadapan. Perlahankan kenderaan dan beri ruang yang mencukupi kepada penunggang basikal.",
            zh=f"前方{name_zh}。请减速，并给骑行者留出足够空间。",
        )
    if action is ActionCode.WATCH_ANIMALS:
        return PhraseText(
            en=f"{name_en} ahead. Ease off the accelerator and watch both sides of the road.",
            ms=f"{name_ms} di hadapan. Lepaskan sedikit pedal minyak dan perhatikan kedua-dua sisi jalan.",
            zh=f"前方{name_zh}。请放慢车速，并观察道路两侧。",
        )
    if action is ActionCode.WATCH_TRAFFIC_SIGNAL:
        return PhraseText(
            en="Traffic signals are ahead. Prepare to stop if the light changes.",
            ms="Lampu isyarat berada di hadapan. Bersedia untuk berhenti jika lampu berubah.",
            zh="前方有交通信号灯。如灯号变化，请准备停车。",
        )
    if action is ActionCode.WATCH_RAILWAY:
        return PhraseText(
            en=f"{name_en} ahead. Slow down, look both ways, and never stop on the crossing.",
            ms=f"{name_ms} di hadapan. Perlahankan kenderaan, lihat kedua-dua arah, dan jangan berhenti di atas lintasan.",
            zh=f"前方{name_zh}。请减速，观察两侧，绝不要停在铁路道口上。",
        )
    if action in {ActionCode.WATCH_ROAD_HAZARD, ActionCode.UNKNOWN_CAUTION}:
        return PhraseText(
            en=f"{name_en} ahead. Slow down and watch the road condition, lane position, and nearby traffic carefully.",
            ms=f"{name_ms} di hadapan. Perlahankan kenderaan dan perhatikan keadaan jalan, kedudukan lorong, serta trafik sekitar.",
            zh=f"前方{name_zh}。请减速，并仔细观察路况、车道位置和周围车辆。",
        )
    if action is ActionCode.REDUCE_SPEED:
        return PhraseText(
            en=f"{name_en} ahead. Reduce speed smoothly and stay prepared to brake or steer gently.",
            ms=f"{name_ms} di hadapan. Kurangkan kelajuan dengan lancar dan bersedia untuk membrek atau mengawal stereng dengan lembut.",
            zh=f"前方{name_zh}。请平稳减速，并准备轻柔刹车或转向。",
        )
    if action is ActionCode.SET_TARGET_SPEED:
        return PhraseText(
            en=f"{name_en} sign ahead. Please read the posted speed and keep your vehicle within the limit.",
            ms=f"Papan tanda {name_ms} di hadapan. Sila baca had laju yang dipaparkan dan pastikan kenderaan mematuhinya.",
            zh=f"前方{name_zh}标志。请留意标示速度，并保持车辆在限速内行驶。",
        )
    if action is ActionCode.HEIGHT_RESTRICTION:
        return PhraseText(
            en=f"{name_en} ahead. Check your vehicle height before passing through.",
            ms=f"{name_ms} di hadapan. Periksa ketinggian kenderaan sebelum melalui laluan ini.",
            zh=f"前方{name_zh}。通过前请确认车辆高度。",
        )
    if action is ActionCode.WIDTH_RESTRICTION:
        return PhraseText(
            en=f"{name_en} ahead. Check your vehicle width and keep centered.",
            ms=f"{name_ms} di hadapan. Periksa lebar kenderaan dan kekalkan posisi di tengah lorong.",
            zh=f"前方{name_zh}。请确认车辆宽度，并保持在车道中央。",
        )
    if action is ActionCode.WEIGHT_RESTRICTION:
        return PhraseText(
            en=f"{name_en} ahead. Check your vehicle load before continuing.",
            ms=f"{name_ms} di hadapan. Periksa muatan kenderaan sebelum meneruskan perjalanan.",
            zh=f"前方{name_zh}。继续前请确认车辆载重。",
        )
    return PhraseText(
        en=f"{name_en} ahead. Use this information early and keep your attention on the road.",
        ms=f"{name_ms} di hadapan. Gunakan maklumat ini lebih awal dan kekalkan perhatian pada jalan.",
        zh=f"前方{name_zh}。请提前利用此信息，并保持注意道路情况。",
    )


def _fallback_text() -> PhraseText:
    return PhraseText(
        en=(
            "A road sign was detected ahead, but I am not confident about its meaning. "
            "Please check the road carefully and drive with caution."
        ),
        ms=(
            "Papan tanda dikesan di hadapan, tetapi maksudnya tidak pasti. "
            "Sila perhatikan jalan dengan teliti dan pandu dengan berhati-hati."
        ),
        zh="前方检测到交通标志，但含义不确定。请仔细观察道路，并谨慎驾驶。",
    )


def _blank_assets(phrase_id: str) -> dict[str, dict[str, Any]]:
    return {
        language: {
            "src": f"/audio/p16/{language}/{phrase_id}.wav",
            "sha256": None,
            "bytes": None,
            "duration_seconds": None,
            "voice": None,
            "generated": False,
        }
        for language in LANGUAGES
    }


def _phrase(
    phrase_id: str,
    text: PhraseText,
    *,
    semantic_sign_id: str | None,
    audio_key: str | None,
    base_action: str,
    severity: str,
    parameter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    priority = PRIORITY_BY_SEVERITY.get(severity, 2)
    return {
        "phrase_id": phrase_id,
        "semantic_sign_id": semantic_sign_id,
        "audio_key": audio_key,
        "base_action": base_action,
        "severity": severity,
        "priority": priority,
        "interrupts_lower_priority": priority >= 3,
        "cooldown_seconds": 8.0 if priority >= 3 else 12.0,
        "parameter": parameter,
        "text": text.model_dump(),
        "assets": _blank_assets(phrase_id),
    }


def build_advisory_manifest() -> dict[str, Any]:
    catalogue = load_catalogue()
    phrases: dict[str, dict[str, Any]] = {}
    semantic_phrase_ids: dict[str, str] = {}
    audio_key_phrase_ids: dict[str, str] = {}

    for entry in sorted(catalogue.entries, key=lambda item: item.semantic_sign_id):
        phrase_id = entry.semantic_sign_id
        phrases[phrase_id] = _phrase(
            phrase_id,
            _generic_entry_text(entry),
            semantic_sign_id=entry.semantic_sign_id,
            audio_key=entry.audio_key,
            base_action=entry.base_action.value,
            severity=entry.severity.value,
        )
        semantic_phrase_ids[entry.semantic_sign_id] = phrase_id
        audio_key_phrase_ids[entry.audio_key] = phrase_id

    phrases["unknown_sign"] = _phrase(
        "unknown_sign",
        _fallback_text(),
        semantic_sign_id="unknown_sign",
        audio_key="unknown_sign",
        base_action=ActionCode.UNKNOWN_CAUTION.value,
        severity=Severity.CAUTION.value,
    )

    speed_variants: dict[str, str] = {}
    minimum_speed_variants: dict[str, str] = {}
    temporary_speed_variants: dict[str, str] = {}
    for value in SPEED_VALUES_KMH:
        value_id = phrase_id_value(value)
        phrase_id = f"speed_limit_{value_id}_kmh"
        speed_variants[str(value)] = phrase_id
        phrases[phrase_id] = _phrase(
            phrase_id,
            _speed_text(value),
            semantic_sign_id="maximum_speed",
            audio_key="maximum_speed",
            base_action=ActionCode.SET_TARGET_SPEED.value,
            severity=Severity.CRITICAL.value,
            parameter={"kind": "speed", "value": value, "unit": "KM/H"},
        )

        minimum_phrase_id = f"minimum_speed_{value_id}_kmh"
        minimum_speed_variants[str(value)] = minimum_phrase_id
        phrases[minimum_phrase_id] = _phrase(
            minimum_phrase_id,
            _speed_text(value, minimum=True),
            semantic_sign_id="minimum_speed",
            audio_key="minimum_speed",
            base_action=ActionCode.SET_TARGET_SPEED.value,
            severity=Severity.WARNING.value,
            parameter={"kind": "speed", "value": value, "unit": "KM/H"},
        )

        temporary_phrase_id = f"temporary_speed_limit_{value_id}_kmh"
        temporary_speed_variants[str(value)] = temporary_phrase_id
        phrases[temporary_phrase_id] = _phrase(
            temporary_phrase_id,
            _speed_text(value, temporary=True),
            semantic_sign_id="temporary_speed_limit",
            audio_key="temporary_speed_limit",
            base_action=ActionCode.SET_TARGET_SPEED.value,
            severity=Severity.CRITICAL.value,
            parameter={"kind": "speed", "value": value, "unit": "KM/H"},
        )

    height_variants: dict[str, str] = {}
    for value in HEIGHT_VALUES_M:
        value_id = phrase_id_value(value)
        phrase_id = f"height_limit_{value_id}_m"
        height_variants[_format_number(value)] = phrase_id
        phrases[phrase_id] = _phrase(
            phrase_id,
            _height_text(value),
            semantic_sign_id="height_restriction",
            audio_key="height_restriction",
            base_action=ActionCode.HEIGHT_RESTRICTION.value,
            severity=Severity.CRITICAL.value,
            parameter={"kind": "height", "value": value, "unit": "M"},
        )

    width_variants: dict[str, str] = {}
    for value in WIDTH_VALUES_M:
        value_id = phrase_id_value(value)
        phrase_id = f"width_limit_{value_id}_m"
        width_variants[_format_number(value)] = phrase_id
        phrases[phrase_id] = _phrase(
            phrase_id,
            _width_text(value),
            semantic_sign_id="width_restriction",
            audio_key="width_restriction",
            base_action=ActionCode.WIDTH_RESTRICTION.value,
            severity=Severity.CRITICAL.value,
            parameter={"kind": "width", "value": value, "unit": "M"},
        )

    weight_variants: dict[str, str] = {}
    for value in WEIGHT_VALUES_T:
        value_id = phrase_id_value(value)
        phrase_id = f"weight_limit_{value_id}_t"
        weight_variants[str(value)] = phrase_id
        phrases[phrase_id] = _phrase(
            phrase_id,
            _weight_text(value),
            semantic_sign_id="weight_restriction",
            audio_key="weight_restriction",
            base_action=ActionCode.WEIGHT_RESTRICTION.value,
            severity=Severity.CRITICAL.value,
            parameter={"kind": "weight", "value": value, "unit": "T"},
        )

    return {
        "schema_version": "p16.advisory_audio.v1",
        "catalogue_version": catalogue.catalogue_version,
        "languages": list(LANGUAGES),
        "description": "Offline human driver-advisory audio phrases for RoadSign Assist.",
        "fallback_phrase_id": "unknown_sign",
        "semantic_phrase_ids": semantic_phrase_ids,
        "audio_key_phrase_ids": audio_key_phrase_ids,
        "variant_phrase_ids": {
            "speed_limit_kmh": speed_variants,
            "minimum_speed_kmh": minimum_speed_variants,
            "temporary_speed_limit_kmh": temporary_speed_variants,
            "height_limit_m": height_variants,
            "width_limit_m": width_variants,
            "weight_limit_t": weight_variants,
        },
        "phrases": dict(sorted(phrases.items())),
    }


def _wave_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return round(frames / float(rate), 3) if rate else None
    except (wave.Error, OSError, EOFError):
        return None


def update_asset_metadata(
    manifest: dict[str, Any],
    public_root: Path,
    voice_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    voices = voice_names or {}
    for phrase in manifest["phrases"].values():
        for language, asset in phrase["assets"].items():
            src = str(asset["src"]).lstrip("/")
            path = public_root / src
            if not path.exists():
                continue
            data = path.read_bytes()
            asset["sha256"] = hashlib.sha256(data).hexdigest()
            asset["bytes"] = len(data)
            asset["duration_seconds"] = _wave_duration(path)
            asset["voice"] = voices.get(language) or asset.get("voice")
            asset["generated"] = True
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

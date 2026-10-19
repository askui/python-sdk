from askui.settings import SETTINGS
from askui.telemetry import Telemetry
from askui.telemetry.processors import Segment

telemetry = Telemetry(SETTINGS.telemetry)

if SETTINGS.telemetry.enabled and SETTINGS.telemetry.segment:
    telemetry.add_processor(Segment(SETTINGS.telemetry.segment))

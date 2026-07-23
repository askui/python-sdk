# Using Agents

AskUI Vision Agent provides four predefined agent types for different automation targets. All agents share the same core API (`act()`, `get()`, `locate()`) but are optimized for their respective platforms. Each agent comes with its own system prompt tailored to its platform-specific tools and capabilities.

## ComputerAgent

Use this agent for desktop automation on Windows, macOS, and Linux. Uses AskUI Agent OS to control mouse, keyboard, and capture screenshots.

```python
from askui import ComputerAgent

with ComputerAgent() as agent:
    agent.act("Open the mail app and summarize all unread emails")
```

**Default tools:** `screenshot`, `mouse_click`, `mouse_move`, `mouse_scroll`, `mouse_hold_down`, `mouse_release`, `type`, `keyboard_tap`, `keyboard_pressed`, `keyboard_release`, `get_mouse_position`, `get_system_info`, `list_displays`, `retrieve_active_display`, `set_active_display`, `list_agent_os_target_computers`, `switch_agent_os_target_computer`, `get_current_computer_target_id`

A single `ComputerAgent` can also drive multiple machines (local and remote) at once. See [Multiple Target Computers](13_multi_target_computers.md).

## AndroidAgent

Use this agent for automation of Android devices via ADB. Supports tapping, swiping, typing, and shell commands.

```python
from askui import AndroidAgent

with AndroidAgent(device=0) as agent:
    agent.tap("Login button")
    agent.swipe(start=(500, 1000), end=(500, 300))
    agent.act("Navigate to settings and enable notifications")
```

Requires the `android` dependency installed (`pip install askui[android]`) and a connected device (physical or emulator).

**Default tools:** `screenshot`, `tap`, `type`, `swipe`, `drag_and_drop`, `key_tap_event`, `key_combination`, `shell`, `select_device_by_serial_number`, `select_display_by_unique_id`, `get_connected_devices_serial_numbers`, `get_connected_displays_infos`, `get_current_connected_device_infos`

### Selecting a display

By default the agent drives the first detected display. On multi-display hardware (e.g. automotive head units) you can pin a specific display with the `display` parameter:

```python
from askui import AndroidAgent
from askui.tools.android.agent_os import AndroidDisplay

# Pin one display by its exact ids (bypasses auto-detection).
# The ids below are placeholders — read your device's real values from
# `adb shell dumpsys display` (see the command at the end of this section).
with AndroidAgent(
    device="emulator-5554",
    display=AndroidDisplay(unique_display_id=1234567890123456789, display_name="secondary", display_id=2),
    display_allow_switching=False,
) as agent:
    agent.act("Open settings")
```

`display` accepts:

- **`AndroidDisplay`** — pins that exact display and bypasses auto-detection, so you control the ids used for shell commands. `display_id` is the logical id passed to `input` (tap/swipe/type) as `-d <id>`; `unique_display_id` is the physical id passed to `screencap` as `-d <id>`. Get both from the device: `adb shell dumpsys display` (look for the `mViewports` line, which maps `displayId ↔ uniqueId`).
- **`list[AndroidDisplay]`** — the authoritative set of selectable displays. The first is active, and the agent may switch among them at runtime with correct ids.
- **`int`** — select by index, **`str`** — select by name (both via auto-detection).

Either id may be `None`, which omits the `-d` flag so that command targets adb's **default display** (display 0). Use `display_id=None` only when your target screen is the default display; on a multi-display setup, pass the real logical id instead.

Set `display_allow_switching=False` to remove the runtime display/device selection tools, so a pinned `display` cannot be changed mid-run by the model.

You can find the ids for a device with (the quotes keep the pipe inside the device shell, so this works the same on PowerShell, cmd, and bash):

```bash
adb shell "dumpsys display | grep -iE 'mViewports|uniqueId'"
```

## WebVisionAgent

For web browser automation using Playwright. Extends `ComputerAgent` with web-specific tools like navigation, URL handling, and page title retrieval.

```python
from askui import WebVisionAgent

with WebVisionAgent() as agent:
    agent.tools.os.goto("https://example.com")
    agent.click("Sign In")
    agent.act("Fill out the contact form and submit")
```

**Default tools:** All `ComputerAgent` tools plus `goto`, `back`, `forward`, `get_page_title`, `get_page_url`

## MultiDeviceAgent

Use this agent when you need to control a desktop computer and an Android device within the same task. The agent has access to both the full set of computer tools (via AskUI Agent OS) and Android tools (via ADB), and can switch between devices seamlessly during execution.

This is useful for cross-device workflows, such as triggering an action on the desktop and verifying the result on a mobile device, or transferring data between devices.

```python
from askui import MultiDeviceAgent

with MultiDeviceAgent(android_device_sn="emulator-5554") as agent:
    agent.act("Open the web app on the computer and send a push notification, then verify it appears on the Android device")
```

If you have multiple Android devices connected, pass the serial number of the target device via `android_device_sn`. You can find serial numbers by running `adb devices`. If omitted, no device is preselected and the agent will select one at runtime.

Requires the `android` dependency installed (`pip install askui[android]`) and a connected device (physical or emulator).

**Default tools:** All `ComputerAgent` tools plus all `AndroidAgent` tools. Additional tools can be provided via the `act_tools` parameter.

## Choosing an Agent

| Target | Agent | Backend |
|--------|-------|---------|
| Desktop (Windows/macOS/Linux) | `ComputerAgent` | AskUI Agent OS (gRPC) |
| Android devices | `AndroidAgent` | ADB |
| Desktop + Android | `MultiDeviceAgent` | AskUI Agent OS (gRPC) + ADB |
| Web browsers | `WebVisionAgent` | Playwright |

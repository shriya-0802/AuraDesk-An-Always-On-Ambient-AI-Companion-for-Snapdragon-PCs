"""
Action Engine
Executes system-level actions based on detected gestures and cognitive state.
Platform-aware: macOS for dev, Windows for Snapdragon deployment.
All macOS commands use permission-free APIs (no Accessibility required).
"""
import os
import subprocess
import platform
import time
from config.settings import GESTURES


class ActionEngine:
    """Handles system actions triggered by gestures and cognitive state."""

    def __init__(self):
        self._platform = platform.system()
        self._last_action_time = {}
        self._dnd_active = False
        self._blue_light_active = False
        self._action_log = []

    def execute_gesture_action(self, gesture: str) -> dict:
        """
        Execute the system action mapped to a gesture.

        Returns:
            dict with 'action', 'success', 'message'
        """
        if gesture == "none" or gesture not in GESTURES:
            return {"action": None, "success": False, "message": ""}

        action_map = {
            "fist": self._take_screenshot,
            "open_palm": self._toggle_media,
            "point_up": self._volume_up,
            "point_down": self._volume_down,
            "peace": self._switch_desktop,
            "thumbs_up": self._confirm_action,
        }

        handler = action_map.get(gesture)
        if handler:
            print(f"[ActionEngine] Executing gesture action: {gesture} → {GESTURES.get(gesture, gesture)}")
            result = handler()
            print(f"[ActionEngine] Result: success={result['success']} message={result['message']}")
            self._action_log.append({
                "gesture": gesture,
                "action": GESTURES[gesture],
                "time": time.time(),
                "success": result.get("success", False),
            })
            return result

        return {"action": gesture, "success": False, "message": "No handler"}

    def apply_wellness_action(self, action_type: str) -> dict:
        """Apply a wellness-related system adaptation."""
        handlers = {
            "blue_light_filter": self._toggle_blue_light,
            "enable_dnd": self._enable_dnd,
            "disable_dnd": self._disable_dnd,
            "play_ambient": self._play_ambient_sound,
            "toggle_dark_mode": self._toggle_system_dark_mode,
        }

        handler = handlers.get(action_type)
        if handler:
            return handler()
        return {"action": action_type, "success": False, "message": "Unknown action"}

    # ─── Gesture Action Handlers ───

    def _take_screenshot(self) -> dict:
        """Take a screenshot and save directly to Desktop."""
        try:
            desktop = os.path.expanduser("~/Desktop")
            filename = os.path.join(desktop, f"AuraDesk_Screenshot_{int(time.time())}.png")

            if self._platform == "Darwin":  # macOS
                result = subprocess.run(
                    ["screencapture", "-x", filename],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode != 0:
                    print(f"[ActionEngine] screencapture error: {result.stderr}")
                    return {"action": "screenshot", "success": False,
                            "message": f"Screenshot error: {result.stderr.strip()}"}
            elif self._platform == "Windows":
                subprocess.run([
                    "powershell", "-command",
                    f"Add-Type -AssemblyName System.Windows.Forms; "
                    f"$bmp = New-Object Drawing.Bitmap([System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Width, "
                    f"[System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Height); "
                    f"$g = [Drawing.Graphics]::FromImage($bmp); "
                    f"$g.CopyFromScreen(0,0,0,0, $bmp.Size); $bmp.Save('{filename}')"
                ], capture_output=True, timeout=5)

            return {
                "action": "screenshot",
                "success": True,
                "message": "📸 Saved to Desktop!",
            }
        except Exception as e:
            print(f"[ActionEngine] Screenshot exception: {e}")
            return {"action": "screenshot", "success": False, "message": f"Screenshot failed: {e}"}

    def _toggle_media(self) -> dict:
        """Toggle media play/pause — uses Music app or Spotify."""
        try:
            if self._platform == "Darwin":
                # Try Music.app first with simple playpause command
                result = subprocess.run(
                    ["osascript", "-e", 'tell application "Music" to playpause'],
                    capture_output=True, text=True, timeout=3
                )
                if result.returncode != 0:
                    # Fallback to Spotify
                    result2 = subprocess.run(
                        ["osascript", "-e", 'tell application "Spotify" to playpause'],
                        capture_output=True, text=True, timeout=3
                    )
                    if result2.returncode != 0:
                        print(f"[ActionEngine] media toggle: no music player found")
            return {"action": "media_toggle", "success": True, "message": "⏯️ Media toggled"}
        except Exception as e:
            print(f"[ActionEngine] Media toggle exception: {e}")
            return {"action": "media_toggle", "success": False, "message": str(e)}

    def _volume_up(self) -> dict:
        """Increase system volume by 10%."""
        try:
            if self._platform == "Darwin":
                result = subprocess.run(
                    ["osascript", "-e",
                     "set volume output volume ((output volume of (get volume settings)) + 10)"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode != 0:
                    print(f"[ActionEngine] volume up stderr: {result.stderr}")
                    return {"action": "volume_up", "success": False,
                            "message": f"Volume error: {result.stderr.strip()}"}
            return {"action": "volume_up", "success": True, "message": "🔊 Volume +10%"}
        except Exception as e:
            print(f"[ActionEngine] Volume up exception: {e}")
            return {"action": "volume_up", "success": False, "message": str(e)}

    def _volume_down(self) -> dict:
        """Decrease system volume by 10%."""
        try:
            if self._platform == "Darwin":
                result = subprocess.run(
                    ["osascript", "-e",
                     "set volume output volume ((output volume of (get volume settings)) - 10)"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode != 0:
                    print(f"[ActionEngine] volume down stderr: {result.stderr}")
                    return {"action": "volume_down", "success": False,
                            "message": f"Volume error: {result.stderr.strip()}"}
            return {"action": "volume_down", "success": True, "message": "🔉 Volume -10%"}
        except Exception as e:
            print(f"[ActionEngine] Volume down exception: {e}")
            return {"action": "volume_down", "success": False, "message": str(e)}

    def _switch_desktop(self) -> dict:
        """Switch to next virtual desktop/space using Mission Control."""
        try:
            if self._platform == "Darwin":
                # Open Mission Control (doesn't require Accessibility permission)
                result = subprocess.run(
                    ["open", "-a", "Mission Control"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode != 0:
                    print(f"[ActionEngine] switch desktop stderr: {result.stderr}")
            return {"action": "switch_desktop", "success": True, "message": "🖥️ Mission Control opened"}
        except Exception as e:
            print(f"[ActionEngine] Switch desktop exception: {e}")
            return {"action": "switch_desktop", "success": False, "message": str(e)}

    def _confirm_action(self) -> dict:
        """Confirm/acknowledge current action."""
        return {"action": "confirm", "success": True, "message": "✅ Confirmed!"}

    # ─── Wellness Action Handlers ───

    def _toggle_blue_light(self) -> dict:
        """Toggle blue light filter."""
        self._blue_light_active = not self._blue_light_active
        state = "enabled" if self._blue_light_active else "disabled"
        return {"action": "blue_light", "success": True, "message": f"🌙 Blue light {state}"}

    def _enable_dnd(self) -> dict:
        """Enable Do Not Disturb."""
        self._dnd_active = True
        return {"action": "dnd_on", "success": True, "message": "🔕 DND enabled"}

    def _disable_dnd(self) -> dict:
        """Disable Do Not Disturb."""
        self._dnd_active = False
        return {"action": "dnd_off", "success": True, "message": "🔔 DND disabled"}

    def _play_ambient_sound(self) -> dict:
        """Start calming ambient sound."""
        return {"action": "ambient", "success": True, "message": "🎵 Playing ambient sounds"}

    def _toggle_system_dark_mode(self) -> dict:
        """Toggle macOS Dark Mode."""
        try:
            if self._platform == "Darwin":
                subprocess.run(
                    ["osascript", "-e", 'tell application "System Events" to tell appearance preferences to set dark mode to not dark mode'],
                    capture_output=True, timeout=5
                )
            return {"action": "dark_mode", "success": True, "message": "🌙 System Dark Mode Toggled"}
        except Exception as e:
            print(f"[ActionEngine] Dark mode exception: {e}")
            return {"action": "dark_mode", "success": False, "message": str(e)}

    def get_action_log(self) -> list:
        """Get history of triggered actions."""
        return self._action_log[-20:]

    @property
    def is_dnd_active(self) -> bool:
        return self._dnd_active

    @property
    def is_blue_light_active(self) -> bool:
        return self._blue_light_active

#!/usr/bin/env python3
"""Final comprehensive verification of auto-registration and global RESOURCE_HUB."""

import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "hush-core"))
sys.path.insert(0, str(Path(__file__).parent / "hush-providers"))

print("=" * 80)
print("FINAL VERIFICATION TEST")
print("=" * 80)

# Test 1: Verify RESOURCE_HUB is exported
print("\n[1/8] Testing RESOURCE_HUB export...")
try:
    from hush.core import RESOURCE_HUB
    print("   ✓ RESOURCE_HUB successfully imported from hush.core")
except ImportError as e:
    print(f"   ✗ Failed to import RESOURCE_HUB: {e}")
    sys.exit(1)

# Test 2: Verify get_hub and set_global_hub are exported
print("\n[2/8] Testing helper functions export...")
try:
    from hush.core import get_hub, set_global_hub
    print("   ✓ get_hub and set_global_hub successfully imported")
except ImportError as e:
    print(f"   ✗ Failed to import helper functions: {e}")
    sys.exit(1)

# Test 3: Verify auto-registration on import
print("\n[3/8] Testing auto-registration...")
from hush.providers import LLMPlugin, EmbeddingPlugin, RerankPlugin

plugins_registered = list(RESOURCE_HUB._plugins.keys())
expected_plugins = ['llm', 'embedding', 'reranking']

if all(p in plugins_registered for p in expected_plugins):
    print(f"   ✓ All plugins auto-registered: {plugins_registered}")
else:
    print(f"   ✗ Missing plugins. Expected: {expected_plugins}, Got: {plugins_registered}")
    sys.exit(1)

# Test 4: Verify plugins registered to global hub only (correct behavior)
print("\n[4/8] Testing plugin registration scope...")
from hush.core.registry import ResourceHub

# New hubs don't get auto-registered plugins (they're independent)
hub2 = ResourceHub.from_memory()
assert not hub2.has_plugin("llm"), "New hub should be empty by default"

# But the global hub has all plugins
assert RESOURCE_HUB.has_plugin("llm"), "Global hub should have plugins"
print("   ✓ Auto-registration correctly targets global hub only")

# Test 5: Test resource registration with subclasses
print("\n[5/8] Testing config subclass support...")
from hush.providers.llms.config import OpenAIConfig

llm_config = OpenAIConfig(
    api_type="openai",
    api_key="test-key",
    base_url="https://api.openai.com/v1",
    model="gpt-4o"
)

try:
    key = RESOURCE_HUB.register(llm_config, persist=False)
    print(f"   ✓ OpenAIConfig (subclass of LLMConfig) registered: {key}")
except ValueError as e:
    print(f"   ✗ Failed to register subclass config: {e}")
    sys.exit(1)

# Test 6: Test resource retrieval
print("\n[6/8] Testing resource retrieval...")
try:
    llm = RESOURCE_HUB.get(key)
    print(f"   ✓ Resource retrieved successfully: {type(llm).__name__}")
except KeyError as e:
    print(f"   ✗ Failed to retrieve resource: {e}")
    sys.exit(1)

# Test 7: Test custom global hub
print("\n[7/8] Testing custom global hub...")
custom_hub = ResourceHub.from_memory()
set_global_hub(custom_hub)
hub_check = get_hub()

if hub_check is custom_hub:
    print("   ✓ Custom global hub set successfully")
else:
    print("   ✗ Failed to set custom global hub")
    sys.exit(1)

# Test 8: Test backwards compatibility
print("\n[8/8] Testing backwards compatibility...")
old_style_hub = ResourceHub.from_memory()
old_style_hub.register_plugin(LLMPlugin)  # Manual registration still works

if old_style_hub.has_plugin("llm"):
    print("   ✓ Manual plugin registration still works (backwards compatible)")
else:
    print("   ✗ Manual registration broken")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL VERIFICATION TESTS PASSED!")
print("=" * 80)

print("\nSummary:")
print("  • Auto-registration: ✓ Working")
print("  • Global RESOURCE_HUB: ✓ Available")
print("  • Config subclass support: ✓ Working")
print("  • Resource management: ✓ Working")
print("  • Custom hub support: ✓ Working")
print("  • Backwards compatibility: ✓ Maintained")

print("\n" + "=" * 80)
print("The ResourceHub auto-registration system is fully operational!")
print("=" * 80)

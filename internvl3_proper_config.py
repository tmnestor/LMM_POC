"""
Proper InternVL3-8B Configuration (Not Emergency Minimal)

Based on analysis of the broadcasting error, the correct configuration
should respect the 1792 vision token limit (7 tiles × 256 tokens/tile).
"""

# PROPER CONFIGURATION - Not emergency minimal!
# Based on 1792 vision token limit = 7 tiles maximum

OPTIMAL_SETTINGS = {
    'input_size': 448,    # Standard resolution
    'max_num': 7,         # 7 tiles × 256 tokens = 1792 tokens (exact limit)
}

CONSERVATIVE_SETTINGS = {
    'input_size': 448,    # Standard resolution
    'max_num': 6,         # 6 tiles × 256 tokens = 1536 tokens (safe margin)
}

HIGHER_RESOLUTION_SETTINGS = {
    'input_size': 560,    # Higher resolution
    'max_num': 5,         # 5 tiles × 256 tokens = 1280 tokens (safe for higher res)
}

# Test these configurations in your notebook:
print("🎯 PROPER INTERNVL3-8B CONFIGURATIONS:")
print()
print("1. OPTIMAL (try first):")
print(f"   load_image(imageName, input_size={OPTIMAL_SETTINGS['input_size']}, max_num={OPTIMAL_SETTINGS['max_num']})")
print()
print("2. CONSERVATIVE (fallback):")
print(f"   load_image(imageName, input_size={CONSERVATIVE_SETTINGS['input_size']}, max_num={CONSERVATIVE_SETTINGS['max_num']})")
print()
print("3. HIGHER RESOLUTION (if you need more detail):")
print(f"   load_image(imageName, input_size={HIGHER_RESOLUTION_SETTINGS['input_size']}, max_num={HIGHER_RESOLUTION_SETTINGS['max_num']})")
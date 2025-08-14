#!/bin/bash
# Clear problematic CUDA environment variables before running Llama processing

echo "🧹 Clearing CUDA environment variables..."

# Unset the problematic PYTORCH_CUDA_ALLOC_CONF if it contains expandable_segments
if [[ "$PYTORCH_CUDA_ALLOC_CONF" == *"expandable_segments"* ]]; then
    echo "⚠️  Found problematic PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
    unset PYTORCH_CUDA_ALLOC_CONF
    echo "✅ Cleared PYTORCH_CUDA_ALLOC_CONF"
fi

# Set safe configuration
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
echo "🔧 Set safe PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"

echo "✅ Environment ready for Llama processing"
echo ""
echo "You can now run:"
echo "  python llama_keyvalue.py"
echo ""
echo "Or source this script in your current shell:"
echo "  source clear_cuda_env.sh"
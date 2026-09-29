#!/usr/bin/env bash
set -euo pipefail

mkdir -p pretrained
curl --fail --location --retry 3 \
  https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth \
  --output pretrained/convnext_tiny_22k_224.pth

echo "b2b2153ae940daa9d60bd75e0ed5065b8ec60bda247941c43bfeaad638068781  pretrained/convnext_tiny_22k_224.pth" \
  | sha256sum --check

echo "ConvNeXt-Tiny ImageNet-22K weights are ready."
echo "CXR-BERT and the Hugging Face ConvNeXt used by GuideDecoder are cached automatically on first use."

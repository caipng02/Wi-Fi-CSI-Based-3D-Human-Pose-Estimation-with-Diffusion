# Source attribution

The Phase 1 files are the native PyTorch ports used in this FYP experiment.

- DT-Pose encoder, pose decoder, pretraining architecture and objectives follow
  cseeyangchen/DT-Pose at commit 2fbbef271454e6fbb95b1a0163b8cdbb96163838.
  https://github.com/cseeyangchen/DT-Pose
- Joint-token global context follows the mmDiff design, adapted from radar tokens
  to CSI patch tokens. The inspected source is FanJunqiao/mmDiff at commit
  29d7df23264e139ca8741945b9394cbf3029d062.
  https://github.com/FanJunqiao/mmDiff
- MM-Fi pairing and split conventions reference ybhbingo/MMFi_dataset at commit
  c9c1e3a8d5099779a618eae1bb351e253daa52d0.
  https://github.com/ybhbingo/MMFi_dataset

The existing FYP model implementations and preprocessing are preserved. Changes
add checkpoint persistence and continuation. The graph adjacency uses a device
aware registered buffer in the existing port. Checkpoint dictionaries from the
FYP port are compatible with this repository. Original upstream DT-Pose and
mmDiff checkpoint layouts require their original loaders or an explicit mapping.

The MM-Fi data remains subject to the authors' terms. Cite the DT-Pose, mmDiff
and MM-Fi papers when reporting work using these implementations or data.

## mmDiff license notice

MIT License

Copyright (c) 2023 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

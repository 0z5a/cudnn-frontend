# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Only explicitly scoped A3 children activate the observe-only logger policy."""

from a3_no_signal_redirect import install

install()

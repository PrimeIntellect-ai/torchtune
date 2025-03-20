# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from torchtune.models.llama_r1._tokenizer import LlamaR1Tokenizer


    
def llama3_r1_tokenizer(
    tokenizer_path: str
):
    return LlamaR1Tokenizer(tokenizer_path=tokenizer_path)

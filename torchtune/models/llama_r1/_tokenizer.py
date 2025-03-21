from torchtune.modules.transforms.tokenizers import HuggingFaceBaseTokenizer, ModelTokenizer
from torchtune.modules.transforms import Transform
from transformers import AutoTokenizer


class LlamaR1Tokenizer(ModelTokenizer, Transform):
    def __init__(self, tokenizer_path):
        self.hf_template_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.hf_tokenizer = HuggingFaceBaseTokenizer(
            tokenizer_json_path=tokenizer_path+"/tokenizer.json",
            tokenizer_config_json_path=tokenizer_path+"/tokenizer_config.json",
            generation_config_path=tokenizer_path+"/generation_config.json"
        )
        self.bos_id = self.hf_tokenizer.bos_id
        self.eos_id = self.hf_tokenizer.eos_id
        self.pad_id = self.hf_tokenizer.pad_id
        
        self.hf_template_tokenizer.chat_template = self.hf_template_tokenizer.chat_template.replace("{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}", "")

        
        
    def encode(self, text, add_bos, add_eos):
        return self.hf_tokenizer.encode(text, add_bos, add_eos)

    def decode(self, token_ids):
        return self.hf_tokenizer.decode(token_ids=token_ids)
    
    def tokenize_messages(self, messages, add_end_tokens):
        messages_proc = [{"role": m.role, "content": m.content[0]["content"]} for m in messages]
        
        chat_tokens = self.hf_template_tokenizer.apply_chat_template(messages_proc, tokenize=True)[:20000]
        mask = [False]*len(chat_tokens)
        mask = mask[:23000]
                               
        return (chat_tokens, mask)
    
    def __call__(
        self, sample, inference: bool = False
    ):
        """
        Apply ``tokenize_messages`` to the "messages" field in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with a "messages" field containing
                a List[Message] to tokenize
            inference (bool): Whether the template is being used for inference or not.

        Returns:
            Mapping[str, Any]: The sample with added "tokens" and "mask" fields
                and the "messages" field removed.
        """
        messages = sample.pop("messages")
        tokens, mask = self.tokenize_messages(messages, add_end_tokens=not inference)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample

        
        
    
        
    
    
    

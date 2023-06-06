from transformers import BertTokenizer, TFBertForMaskedLM
import tensorflow as tf
import asyncio
from cachetools import TTLCache


cache = TTLCache(maxsize=100, ttl=300)
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForMaskedLM.from_pretrained(model_name)


async def getPredict_cached(input_text):
    if input_text in cache:
        return cache[input_text]

    result = await getPredict(input_text)

    cache[input_text] = result

    return result


async def getPredict(text):
    tokens = tokenizer.encode_plus(text, add_special_tokens=True, padding='longest',
                                   max_length=128, truncation=True, return_tensors='tf')

    output = model.predict(tokens.input_ids)
    logits = output.logits
    predictions = tf.argmax(logits, axis=-1).numpy().tolist()[0]
    predicted_token = tokenizer.decode(predictions)
    return predicted_token

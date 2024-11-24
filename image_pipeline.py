import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForCausalLM
from article_scraper import parse_article
from helper import update_row_by_id,get_row_by_id,save_dataframe
from mmkg_ontology import DepictionHandler
import requests
import pandas as pd

def init_florence():
    
    checkpoint='microsoft/Florence-2-base'
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model,processor

def run_captioning(model,processor,task_prompt, image, text_input=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(device),
      pixel_values=inputs["pixel_values"].to(device),
      max_new_tokens=1024,
      num_beams=5,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    print(task_prompt,':',parsed_answer)
    return parsed_answer

# def caption_images(article_link,docs_df,schema,model,processor):
#   article=parse_article(article_link)
#   row=get_row_by_id(docs_df,article_link)
#   if (row is not None) & (row['image-captioning-status']==None) | (row['image-captioning-status']==False):
#     for image in  article['images']:
#       image_url=image['url']
#       caption=image['caption']
#       try:
#         image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
#         generated_caption= run_captioning(model,processor,'<DETAILED_CAPTION>',image)
#         depiction_handler = DepictionHandler(schema) ## try to parse as arg
#         print(f"url:{image_url} caption: {caption}  generated caption:{generated_caption}")
#         depiction_handler.add_image_caption(image_url,caption,generated_caption)
#       except Exception as e:
#         print(e)
#     schema.save(format='ttl')
#     docs_df=update_row_by_id(docs_df,article_link,{'image-captioning-status':True}) # update
#   return docs_df

def caption_images(article_link, docs_df, schema, model, processor):
    article = parse_article(article_link)
    row = get_row_by_id(docs_df, article_link)
    
    if row is not None and (pd.isna(row['image-captioning-status']) or row['image-captioning-status'] == False):
        for image in article['images']:
            image_url = image['url']
            caption = image['caption']
            try:
                image_obj = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
                generated_caption = run_captioning(model, processor, '<DETAILED_CAPTION>', image_obj)
                schema.init_current_article(article_link)
                depiction_handler = DepictionHandler(schema)
                print(f"url: {image_url} caption: {caption}  generated caption: {generated_caption}")
                depiction_handler.add_image_caption(image_url, caption, generated_caption)
            except Exception as e:
                print(f"Error processing image at {image_url}: {e}")
        
        schema.save(format='ttl')
        docs_df = update_row_by_id(docs_df, article_link, {'image-captioning-status': True})  # Update the status
    return docs_df

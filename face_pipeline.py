import io, requests,datetime, uuid, json, PIL
from collections import Counter
import numpy as np 
from PIL import Image, UnidentifiedImageError
from deepface import DeepFace
import matplotlib.pyplot as plt
from pinecone import Pinecone, ServerlessSpec
from google.colab import userdata
from sklearn.decomposition import PCA
from helper import append_rows_to_csv,check_file_exists,initialize_csv_with_headers,read_rows_from_csv

search_engine_id= '502c6eae9f8444e81'
google_search_key=userdata.get('GOOGLE_CUSTOM_SEARCH')
headers = {
        'User-Agent': 'MMKG_DataCrawler/0.1 (uA5N7w2KEk@protonmail.com)'
    }

def init_pinecone_index():
    pc = Pinecone(api_key='67db46e1-1b5e-4138-b34c-77e6f5f73cf5')
    index_name = 'mmkg-index'
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=128,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    index= pc.Index(index_name)
    return index

# def get_google_image_metadata(query, k):
#     url = 'https://www.googleapis.com/customsearch/v1'
#     params = {
#         'q': query,
#         'cx': search_engine_id,
#         'searchType': 'image',
#         'num': k,
#         'rights': 'cc_publicdomain,cc_attribute,cc_sharealike,cc_noncommercial',
#         'key': google_search_key,
#         'fields': 'items(title,link,image/height,image/width)',
#         'imgType':'face' ### Faces
#     }

#     response = requests.get(url, params=params)

#     if response.status_code == 200:
#         data = response.json()
#         items=[]
#         for i, item in enumerate(data['items']):
#             items.append({'image_url' :item['link'],'image_title':item['title']})
#         return {'label':query,'items':items}
#     else:
#         print(f"Failed to search Google images: {response.status_code}")

def get_google_image_metadata(query, k, max_retries=3):
    url = 'https://www.googleapis.com/customsearch/v1'
    valid_images = []
    retries = 0
    start_index = 1
    headers = {
        'User-Agent': 'MMKG_DataCrawler/0.1 (uA5N7w2KEk@protonmail.com)'
    }

    while len(valid_images) < k and retries < max_retries:
        params = {
            'q': query,
            'cx': search_engine_id,
            'searchType': 'image',
            'num': min(k - len(valid_images), 10),
            'rights': 'cc_publicdomain,cc_attribute,cc_sharealike,cc_noncommercial',
            'key': google_search_key,
            'fields': 'items(title,link,image/height,image/width)',
            'imgType': 'face',
            'start': start_index
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'items' in data:
                for item in data['items']:
                    image_url = item['link']
                    
                    # Verify if the image can be fetched
                    try:
                        img_response = requests.get(image_url, stream=True, headers=headers)
                        img_response.raise_for_status()  # Check for valid response
                        Image.open(img_response.raw).convert('RGB')
                        # Image is fetchable, add it to valid_images
                        valid_images.append({'image_url': image_url, 'image_title': item['title']})
                        
                        if len(valid_images) >= k:
                            break  # Stop once we have enough valid images

                    except Exception as e:
                        print(f"Image at {image_url} could not be fetched: {e}")

                start_index += 10  # Move to the next page if needed
            else:
                break  # No more results available
        else:
            print(f"Failed to fetch image metadata: {response.status_code}")
            break

        retries += 1

    return {'label': query, 'items': valid_images[:k]}

def get_embeddings(image_path):
    embedding = DeepFace.represent(img_path=image_path, model_name='Facenet',detector_backend='yolov8')
    return np.array(embedding)

def remove_outliers_mad(embeddings, threshold=6):
    median_embedding = np.median(embeddings, axis=0)
    absolute_deviations = np.abs(embeddings - median_embedding)
    mad = np.median(absolute_deviations, axis=0)
    scaled_mad = mad * 1.4826
    threshold_value = threshold * scaled_mad
    outliers = np.any(absolute_deviations > threshold_value, axis=1)
    return embeddings[~outliers], outliers

def process_images(dataset,threshold=6):
    headers = {
        'User-Agent': 'MMKG_DataCrawler/0.1 (uA5N7w2KEk@protonmail.com)'
    }
    
    filtered_data = []

    for data in dataset:
        embeddings = []
        image_metadata = []
        label = data['label']
        
        for item in data['items']:
            image_url = item['image_url']
            image_title = item['image_title']

            try:
                # Fetch image and convert to numpy array
                response = requests.get(image_url, stream=True, headers=headers)
                response.raise_for_status()  # Ensure we raise an error for bad responses

                image = Image.open(response.raw).convert('RGB')
                embedding_out = get_embeddings(np.asarray(image))

                if len(embedding_out)==1:  # Check if single face  # ? add all faces
                    embeddings.append(embedding_out[0]['embedding'])
                    image_metadata.append({
                        'image_title': image_title,
                        'image_url': image_url,
                        'image_embedding': embedding_out[0]['embedding']
                    })

            except Exception as e:
                print(f"Error processing image from {image_url}: {e}")

        embeddings = np.array(embeddings)
        # Remove outliers using MAD
        filtered_embeddings, outliers = remove_outliers_mad(embeddings,threshold)

        filtered_items = [item for i, item in enumerate(image_metadata) if not outliers[i]]

        filtered_data.append({
            'label': label,
            'items': filtered_items
        })

    return filtered_data

def get_google_image_metadata_and_embeddings(query, k=10, max_retries=5):
    url = 'https://www.googleapis.com/customsearch/v1'
    valid_images = []
    retries = 0
    start_index = 1
    headers = {
        'User-Agent': 'MMKG_DataCrawler/0.1 (uA5N7w2KEk@protonmail.com)'
    }

    while len(valid_images) < k and retries < max_retries:
        params = {
            'q': query,
            'cx': search_engine_id,
            'searchType': 'image',
            'num': min(k - len(valid_images), 10),
            'rights': 'cc_publicdomain,cc_attribute,cc_sharealike,cc_noncommercial',
            'key': google_search_key,
            'fields': 'items(title,link,image/height,image/width)',
            'imgType': 'face',
            'start': start_index
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'items' in data:
                for item in data['items']:
                    image_url = item['link']
                    image_title = item['title']

                    try:
                        img_response = requests.get(image_url, stream=True, headers=headers)
                        img_response.raise_for_status()
                        image = Image.open(img_response.raw).convert('RGB')

                        # Embed the image directly and check for a single face
                        embedding_out = get_embeddings(np.asarray(image))
                        if len(embedding_out) == 1:  # Single face detected
                            valid_images.append({
                                'image_title': image_title,
                                'image_url': image_url,
                                'image_embedding': embedding_out[0]['embedding']
                            })

                            if len(valid_images) >= k:
                                break  # Stop once we have enough valid images

                    except Exception as e:
                        print(f"Error processing image at {image_url}: {e}")

                start_index += 10  # Move to the next page for more results if needed
            else:
                break  # No more results available
        else:
            print(f"Failed to fetch image metadata: {response.status_code}")
            break

        retries += 1

    return {'label': query, 'items': valid_images[:k]}

def upsert_images_to_pinecone(index, namespace, images, batch_size=10):
    vectors = []
    batch_counter = 0
    for data in images:
        label = data['label']
        for item in data['items']:
            image_url = item['image_url']
            image_title = item['image_title']
            embedding=item['image_embedding']
            unique_id = str(uuid.uuid4())
            vectors.append({
                    "id": unique_id,
                    "values": embedding,
                    "metadata": {
                        "label": label,
                        "image_url": image_url,
                        "image_title": image_title
                    }
            })
            # Upsert in batches
            if len(vectors) >= batch_size:
                index.upsert(vectors=vectors, namespace=namespace)
                print(f'Upserting vectors.. batch: {batch_counter}, size: {len(vectors)}')
                vectors = []  # Reset the list after upserting
                batch_counter += 1

    # Upsert any remaining vectors
    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)

def get_top_k_persons(data, k):
    persons = [{'name':entry['name'],'mentions': entry['mentions']} for entry in data if entry['label'] == 'person']
    sorted_persons = sorted(persons, key=lambda x: x['mentions'], reverse=True)
    return sorted_persons[:k]

def get_top_persons_by_mentions(data, min_mentions=3):
    persons = [{'name': entry['name'], 'mentions': entry['mentions']} 
               for entry in data if entry['label'] == 'person' and entry['mentions'] >= min_mentions]
    sorted_persons = sorted(persons, key=lambda x: x['mentions'], reverse=True)
    return sorted_persons

def init_face_label_file(file_path,header=['label', 'n_images', 'timestamp']):
  if not check_file_exists(file_path):
      print(f"{file_path} does not exists.")
      initialize_csv_with_headers(file_path,header)
      append_rows_to_csv(file_path,[['placeholder-label','placeholder-images','placeholder-timestamp']])
  else:
    print(f"{file_path} exists.")

def generate_face_dataset(entities_with_ids, file_path, min_mentions=3):
    entity_image_dataset = []
    top_persons = get_top_persons_by_mentions(entities_with_ids, min_mentions)

    for person in top_persons:
        entries = read_rows_from_csv(file_path)
        if not any(person['name'] in sublist[0] for sublist in entries):
            entity_image_data = get_google_image_metadata_and_embeddings(person['name'], 10, max_retries=5)
            entity_image_dataset.append(entity_image_data)
            append_rows_to_csv(file_path, [[
                f"{person['name']}", f"{len(entity_image_data['items'])}", f"{datetime.datetime.now()}"
            ]])

    return entity_image_dataset

# def generate_face_dataset(entities_with_ids,file_path):
#     entity_image_dataset=[]
#     for e in get_top_k_persons(entities_with_ids,5):  # change 5 to dynamically decide
#         entries= read_rows_from_csv(file_path)
#         if not any(sublist[0] == e['name'] for sublist in entries):
#             entity_image_data=get_google_image_metadata(e['name'] , 10)
#             entity_image_dataset.append(entity_image_data)
#             append_rows_to_csv(file_path,[[f"{e['name']}",f"{len(entity_image_data['items'])}",f"{datetime.datetime.now()}"]])
#     return entity_image_dataset

def process_faces(article,entity_linker,depiction_handler,index,entity_match_threshold=0.7):
    for image in article['images']:
        try:
            # Fetch image and convert to numpy array
            url=image['url']
            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()  # Ensure we raise an error for bad responses
            depiction_image = depiction_handler.add_image(url.split('/')[-1],url) # Add image to schema
            image = Image.open(response.raw).convert('RGB')
            plt.imshow(image)
            plt.show()
            embedding_out = get_embeddings(np.asarray(image))
            for embedding in embedding_out:
                result = index.query(
                    namespace='face-reid',
                    vector=embedding['embedding'],
                    top_k=5,
                    include_values=True,
                    include_metadata=True
                )
                counter=Counter()
                for match in result['matches']:
                    score=match['score']
                    if score > 0.55:
                        metadata=match['metadata']
                        label= metadata['label']
                        counter[label]+=1
                if counter:
                    target_label = counter.most_common(1)[0][0]
                    entity=entity_linker.match_entity(target_label,entity_match_threshold) # Match target entity to schema
                    if entity is not None:
                        print('target label:',target_label,'target_entity:',entity)
                        depiction_handler.add_depiction(entity,depiction_image,str(embedding['facial_area'])) # Add depiction to schema
                    else:
                        print(f"***FACE MATCH {target_label} COULD NOT BE MATCHED TO SCHEMA***")
                else :
                    print('no face match')
        except Exception as e:
            print(f"Error processing image from {url}: {e}")





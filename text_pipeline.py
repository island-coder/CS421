import spacy
from spacy.tokens import Doc

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import requests,json

def resolve_references(doc: Doc) -> str:
    """Function for resolving references with the coref ouput
    doc (Doc): The Doc object processed by the coref pipeline
    RETURNS (str): The Doc string with resolved references
    """
    # token.idx : token.text
    token_mention_mapper = {}
    output_string = ""
    clusters = [
        val for key, val in doc.spans.items() if key.startswith("coref_cluster")
    ]

    # Iterate through every found cluster
    for cluster in clusters:
        first_mention = cluster[0]
        # Iterate through every other span in the cluster
        for mention_span in list(cluster)[1:]:
            # Set first_mention as value for the first token in mention_span in the token_mention_mapper
            token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_

            for token in mention_span[1:]:
                # Set empty string for all the other tokens in mention_span
                token_mention_mapper[token.idx] = ""

    # Iterate through every token in the Doc
    for token in doc:
        # Check if token exists in token_mention_mapper
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        # Else add original token text
        else:
            output_string += token.text + token.whitespace_

    return output_string

def init_rebel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_rebel='Babelscape/rebel-large'

    tokenizer_rebel = AutoTokenizer.from_pretrained(checkpoint_rebel)
    model_rebel = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_rebel)
    model_rebel.to(device)
    return tokenizer_rebel,model_rebel

def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets

def get_wikidata_entity(entity_label):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": entity_label
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data is not None and data["search"]:
            return data["search"][0]["id"]
    return None

def get_dbpedia_entity(entity_label):
    url = "https://lookup.dbpedia.org/api/search?"
    params = {
        "format": "JSON",
        "query": entity_label,
        "maxResults":1
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data is not None and len(data['docs'])>0:
            element=data['docs'][0]
            if float(element['score'][0])>1500:
                #print(f"id:{ element['id'] } score:{element['score']}" )
                return element['id']
    return None

def get_entity_id(label):
        wikidata_id = get_wikidata_entity(label)
        return wikidata_id if wikidata_id is not None else get_dbpedia_entity(label)

def get_coarse_grained_entity_type(entity_id):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": entity_id,
        "props": "claims",
        "languages": "en"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "entities" in data and entity_id in data["entities"]:
            claims = data["entities"][entity_id]["claims"]
            if "P31" in claims:  # P31 is the 'instance of' property
                instance_of = claims["P31"][0]["mainsnak"]["datavalue"]["value"]["id"]
                return get_label_from_id(instance_of)
    return None

def get_label_from_id(entity_id):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": entity_id,
        "languages": "en",
        "props": "labels"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "entities" in data and entity_id in data["entities"]:
            return data["entities"][entity_id]["labels"]["en"]["value"]
    return None

gen_kwargs = {
    "max_length": 512,
    "length_penalty": 0,
    "num_beams": 5,
    "num_return_sequences":4,
}

def predict_triples(sent,tokenizer,model):
    model_inputs = tokenizer(sent, max_length=512, padding=True, truncation=True, return_tensors='pt')
    generated_tokens = model.generate(
        model_inputs["input_ids"].to(model.device),
        attention_mask=model_inputs["attention_mask"].to(model.device),
        **gen_kwargs
    )
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    for sentence in decoded_preds:
        triplets = extract_triplets(sentence)
        for triplet in triplets:
            if triplet['head']:
                #triplet['head'] = {'label': triplet['head'], 'id': get_entity_id(triplet['head'])}
                triplet['head'] = {'label': triplet['head']}
            if triplet['tail']:
                #triplet['tail'] = {'label': triplet['tail'], 'id': get_entity_id(triplet['tail'])}
                triplet['tail'] = {'label': triplet['tail']}
        return triplets
    
def get_top_k_entities(data, k, include:list=None, exclude:list=None):
    # Handle case where both include and exclude are None: include all labels
    if include is None and exclude is None:
        persons = [{'name': entry['name'], 'mentions': entry['mentions']} for entry in data]
    else:
        persons = [{'name': entry['name'], 'mentions': entry['mentions']} 
                   for entry in data 
                   if (include is None or entry['label'] in include) and 
                      (exclude is None or entry['label'] not in exclude)]
    
    # Sort the persons list by mentions in descending order
    sorted_persons = sorted(persons, key=lambda x: x['mentions'], reverse=True)
    
    # Return the top k persons
    return sorted_persons[:k]


# Function to extract triplets using GPT-4o with JSON schema
def extract_triplets(client,model,text):
    # Define the JSON schema for a response format that includes an array of triplets
    json_schema = {
        "name": "relationship_schema",
        "schema": {
            "type": "object",
            "properties": {
                "triplets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "head": {
                                "type": "string",
                                "description": "The head entity in the relationship"
                            },
                            "type": {
                                "type": "string",
                                "description": "The type of relationship"
                            },
                            "tail": {
                                "type": "string",
                                "description": "The tail entity in the relationship"
                            }
                        },
                        "required": ["head", "type", "tail"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["triplets"],
            "additionalProperties": False
        }
    }

    # Create a chat completion request with JSON schema
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert at extracting meaningful relationships from text and formatting them as JSON triplets. Each triplet should follow the order head,type and tail and the schema specified. 

                - Ensure that both "head" and "tail" are specific instances of the following first-class categories: Person, Organization, Event, Geopolitical Entity (GPE), Geographic Location, Nationality, Facility, or Product.
                - The "type" should be a relevant, concise, and context-appropriate relationship between the "head" and "tail" entities.
                - Try to keep the type to one or a maximum of two words. eg:- (attended, associated with,organized by,allies,conflict,...  ect. )
                - Avoid generating duplicate triplets.
                - Use clear and consistent terminology, and prioritize high-quality, contextually accurate triplets.

                Respond with only the JSON output of the extracted triplets.
                """
            },
            {
                "role": "user",
                "content": f"Extract triplets from this text: \"{text}\""
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": json_schema
        }
    )
    # Parse the response
    output = response.choices[0].message.content
    try:
        data = json.loads(output)
        triplets = data.get("triplets", [])
        return triplets
    except json.JSONDecodeError:
        print("Could not parse the response. Ensure the model's output adheres to the schema.")
        return []

from collections import Counter

label_full_form = {
        'EVENT': 'event',
        'FAC': 'facility',
        'GPE': 'geopolitical entity',
        'LANGUAGE': 'language',
        'LAW': 'law',
        'LOC': 'geographic location',
        'NORP': 'nationalities or religious or political groups',
        'ORG': 'organization',
        'PERSON': 'person',
        'PRODUCT': 'product',
        'WORK_OF_ART': 'work of art'
    }

def explain_label(label):
    """Converts the abbreviated NER label into its full form for similarity comparison"""
    return label_full_form.get(label, label)

def get_entities_with_ids(doc):
    unique_entities = set()
    entity_counts = Counter()    
    for ent in doc.ents:
        entity_counts[ent.text]+=1
        unique_entities.add((ent.text, ent.label_))

    entities_with_ids=[]


    for ent in unique_entities:
        text, label_ = ent
        if label_ in ['EVENT','FAC','GPE', 'LANGUAGE', 'LAW', 'LOC', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART'] :
            ent_id = get_entity_id(text)
            entities_with_ids.append({'name':text,'label':explain_label(label_),'id':ent_id,'mentions':entity_counts[text]})
    return entities_with_ids

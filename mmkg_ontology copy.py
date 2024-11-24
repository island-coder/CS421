
import uuid
from rdflib import ConjunctiveGraph,Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD
from rdflib.namespace import split_uri
from rdflib.plugins.stores.memory import Memory
import spacy
from helper import check_file_exists
from text_pipeline import get_wikidata_entity,get_coarse_grained_entity_type

nlp = spacy.load('en_core_web_md')

class RDFSchema:
    def __init__(self, namespace, graph:ConjunctiveGraph,file_path):
        self.graph = graph
        self.namespace = namespace
        self.classes = set()
        self.entities = set()
        self.identifer_prop = None
        self.name_prop = None

        self.article_class=None
        self.article_source_prop=None
        self.has_reference_rel=None
        self.current_article=None

        self.graph.bind("", namespace)
        self.file_path=file_path

    def generate_unique_uri(self):
        unique_id = uuid.uuid4()
        return URIRef(self.namespace + str(unique_id))

    def add_class(self, class_name, superclass=None):
        class_uri = URIRef(self.namespace + class_name.lower().replace(' ', '_'))
        self.graph.add((class_uri, RDF.type, RDFS.Class))
        if superclass:
            superclass_uri = URIRef(self.namespace + superclass.lower().replace(' ', '_'))
            self.graph.add((class_uri, RDFS.subClassOf, superclass_uri))
        self.classes.add(class_uri)
        return class_uri

    def add_property(self, property_name, domain: URIRef, range_: URIRef):
        property_uri = URIRef(self.namespace + property_name.lower().replace(' ', '_'))
        self.graph.add((property_uri, RDF.type, RDF.Property))
        self.graph.add((property_uri, RDFS.domain, domain))
        self.graph.add((property_uri, RDFS.range, range_))
        return property_uri

    def has_property(self, class_uri: URIRef, property_uri: URIRef):
        if (property_uri, RDFS.domain, class_uri) in self.graph:
            return True
        for superclass_uri in self.graph.objects(property_uri, RDFS.domain):
            if (class_uri, RDFS.subClassOf, superclass_uri) in self.graph:
                return True
        return False

    def add_entity(self, class_uri: URIRef, entity_name=None, entity_identifier=None):
        entity_uri = self.generate_unique_uri()
        self.graph.add((entity_uri, RDF.type, class_uri))
        self.entities.add(entity_uri)
        if entity_identifier != None:
             self.graph.add((entity_uri, self.identifer_prop, Literal(entity_identifier)))
        if self.has_property(class_uri,  self.name_prop) and entity_name != None:
            self.graph.add((entity_uri,  self.name_prop , Literal(entity_name)))
        return entity_uri

    def add_property_instance(self, entity_uri: URIRef, property_uri: URIRef, property_value):
        self.graph.add((entity_uri, property_uri, property_value))

    def init_current_article(self,source_url:str):
        article_uri=self.add_entity(self.article_class)
        self.add_property_instance(article_uri,self.article_source_prop,URIRef(source_url))
        self.current_article=article_uri
        return article_uri

    def save(self,format="turtle"):
        return self.graph.serialize(format=format,destination=self.file_path)


class EntityLinker:
    def __init__(self, schema, nlp, source=None):
        self.schema = schema
        self.nlp = nlp
        self.source=source
        self.sub_graph=None
        if source!=None:
          source_uri = URIRef(source)
          self.sub_graph=Graph(store=schema.graph.store,identifier=source_uri)

    def map_ner_entity(self, entity_from_ner, threshold=0.7):
        """Maps entity extracted by NER to entity class from ontology"""
        entity_name, ner_class, entity_identifier = entity_from_ner
        entity_uri = self.match_entity(entity_name, threshold=0.7) ## check if entity already exists

        if entity_uri is None and ner_class is not None:
            ner_entity_doc = self.nlp(ner_class.lower())
            best_match = max(
                self.schema.classes,
                key=lambda schema_class: ner_entity_doc.similarity(self.nlp(self.get_normal_fragment(schema_class))),
                default=None
            )
            
            if best_match:
                best_similarity = ner_entity_doc.similarity(self.nlp(self.get_normal_fragment(best_match)))
                print(f"class:{ner_class} class with best match: {best_match} thresh: {best_similarity}")
                if best_similarity >= threshold:
                    entity_uri = self.schema.add_entity(best_match, entity_name, entity_identifier)
                    print(f"sucessfully mapped entity with name :{entity_name}")
                else:
                    print(f"failed to map entity : {entity_name} with class: {ner_class}")
                    return None
        if entity_uri is not None:
            #add article reference
            self.schema.graph.add((entity_uri,self.schema.has_reference_rel,self.schema.current_article))   
        return entity_uri

    def match_entity(self, target_name, threshold=0.7):
        """Matches entity to ontology by URI using similarity approach with a threshold"""
        target_doc = self.nlp(target_name.lower())
        
        best_match = max(
            self.schema.entities,
            key=lambda uri: target_doc.similarity(self.nlp(str(self.schema.graph.value(uri, self.schema.name_prop)).lower())),
            default=None
        )
        
        if best_match:
            best_similarity = target_doc.similarity(self.nlp(str(self.schema.graph.value(best_match, self.schema.name_prop)).lower()))
            if best_similarity < threshold:
                print(f"failed to find matching entity for : {target_name} , current match: {best_match} with threshold: {best_similarity}")
                return None

        return best_match

    def match_class(self, target_entity_uri):
        """Matches target entity to its class in the ontology"""
        return self.schema.graph.value(target_entity_uri, RDF.type)

    def match_property(self, target_head_class, target_property, target_tail_class=None, threshold=0.7,show_propery_matches=False):
        """Matches property of a class to ontology with a similarity threshold"""
        class_relationships = {s for s, p, o in self.schema.graph.triples((None, RDF.type, RDF.Property))
                            if (s, RDFS.domain, target_head_class) in self.schema.graph and (s, RDFS.range, target_tail_class) in self.schema.graph}
        if show_propery_matches:
            print(class_relationships)
        if class_relationships:
            target_property_doc = self.nlp(target_property.lower())
            best_match = max(
                class_relationships, 
                key=lambda x: target_property_doc.similarity(self.nlp(self.get_normal_fragment(x))),
                default=None
            )

            if best_match:
                best_similarity = target_property_doc.similarity(self.nlp(self.get_normal_fragment(best_match)))
                if best_similarity < threshold:
                    return None

            return best_match
        
        return None

    def map_all_entites_from_ner(self,entites_from_ner,threshold=0.7):
          for e in entites_from_ner:
            self.map_ner_entity(e,threshold)

    def map_triple(self,triple,threshold_entities=0.7,threshold_property=0.7,show_propery_matches=False,show_triples=True):
          head,_type,tail= triple.values()
          head_entity= self.match_entity(head['label'],threshold_entities)
          tail_entity= self.match_entity(tail['label'],threshold_entities)

          if head_entity==None:
            label=head['label']
            id=get_wikidata_entity(label)
            head_entity_= [label,get_coarse_grained_entity_type(id),id]
            print('head entity not found, linking results: ',head_entity_)
            head_entity = self.map_ner_entity(head_entity_)
          if tail_entity==None:
            label=tail['label']
            id=get_wikidata_entity(label)
            tail_entity_ = [label,get_coarse_grained_entity_type(id),id]
            print('tail entity not found, linking results: ',tail_entity_)
            tail_entity = self.map_ner_entity(tail_entity_)

          head_class=self.match_class(head_entity)
          tail_class=self.match_class(tail_entity)
          matched_property=self.match_property(head_class,_type,tail_class,threshold_property,show_propery_matches)
          
          if not matched_property: # try reverse match
            print("trying for flipped match property")
            head_class,tail_class=tail_class,head_class
            matched_property=self.match_property(head_class,_type,tail_class,threshold_property,show_propery_matches)
            # if no match?

          if show_triples:
            print(f" {triple} | {head_class} | {tail_class} | {matched_property} ")

          if head_class!=None and tail_class!=None and matched_property!=None:
            if self.sub_graph!=None:
              self.sub_graph.add((head_entity,matched_property,tail_entity))
            else:
              self.schema.graph.add((head_entity,matched_property,tail_entity))

    @staticmethod
    def get_fragment(uri):
        return split_uri(uri)[1]

    @staticmethod
    def get_normal_fragment(uri):
        return EntityLinker.get_fragment(uri).replace('_',' ')


class DepictionHandler:
    def __init__(self, schema):
        self.schema = schema

    def check_image_exists(self,url: str):
        # Check if an image with the given URL already exists
        for s, p, o in self.schema.graph.triples((None, self.schema.image_url, URIRef(url))):
            return s  # Return the existing image entity
        # print("Image is not in schema")
        return None
        
    #add image
    def add_image(self, filename: str, url: str):
        # Check if an image with the given URL already exists
        image=self.check_image_exists(url)
        if image is None:
            # If no existing image is found, create a new one
            image = self.schema.add_entity(self.schema.image_class)
            self.schema.add_property_instance(image, self.schema.image_url, URIRef(url))
            self.schema.add_property_instance(image, self.schema.image_filename, Literal(filename))
        self.schema.graph.add((image,self.schema.has_reference_rel,self.schema.current_article))    #reference to article
        return image
    
    def add_image_caption(self,url: str,caption=None,generated_caption=None):
        image=self.check_image_exists(url)
        if image is not None:
            if caption is not None:
                self.schema.add_property_instance(image, self.schema.image_caption, Literal(caption))
            if generated_caption is not None:
                self.schema.add_property_instance(image, self.schema.image_generated_caption, Literal(generated_caption))  
        return image

    #add depiction and map entity
    def add_depiction(self, target_entity_uri: URIRef, image_uri: URIRef,bounding_box:str):
        depiction_uri = self.schema.add_entity(self.schema.depiction_class)
        self.schema.graph.add((depiction_uri, self.schema.has_image, image_uri))
        self.schema.graph.add((target_entity_uri, self.schema.has_depiction, depiction_uri))
        self.schema.add_property_instance(depiction_uri, self.schema.bounding_box, Literal(bounding_box))
        #self.schema.graph.add((depiction_uri,self.schema.has_reference_rel,self.schema.current_article))    #reference to article

def init_mmkg(file_path):

    cckg = Namespace("http://demo-mmkg.org/#")
    
    if check_file_exists(file_path):
        g = ConjunctiveGraph()
        g.parse(file_path)
    else:
        g = ConjunctiveGraph()

    # Initialize the schema
    schema = RDFSchema(cckg, g,file_path)

    # Add classes/entity types
    concept_class = schema.add_class("Concept")
    article_class = schema.add_class("Article", superclass="Concept")
    schema.article_class=article_class ###
    person_class = schema.add_class("Person", superclass="Concept")
    organization_class = schema.add_class("Organization", superclass="Concept")
    event_class = schema.add_class("Event", superclass="Concept")
    gpe_class = schema.add_class("Geopolitical entity", superclass="Concept")
    location_class = schema.add_class("Geographic Location", superclass="Concept")
    nationality_group = schema.add_class("Nationality", superclass="Concept") 
    facility_class=schema.add_class("Facility",superclass="Concept")
    product_class =schema.add_class("Product",superclass="Concept")
    schema.image_class = schema.add_class("Image")
    schema.depiction_class = schema.add_class("Depiction")

    # Add properties
    schema.identifer_prop=schema.add_property("has id", concept_class, XSD.string)
    schema.name_prop = schema.add_property("has Name or Title", concept_class, XSD.string)
    schema.add_property("has Role", person_class, XSD.string)
    schema.add_property("has Type", organization_class, XSD.string)
    schema.add_property("has Location", organization_class, location_class)
    schema.add_property("has Date", event_class, XSD.dateTime)
    schema.add_property("has Location", event_class, location_class)
    schema.add_property("has Coordinates", location_class, XSD.string)
    schema.add_property("has PublicationDate", article_class, XSD.dateTime)
    schema.add_property("has Author", article_class, person_class)
    schema.article_source_prop=schema.add_property("has source url", article_class, XSD.anyURI)

    schema.image_url = schema.add_property("has source url", schema.image_class, XSD.anyURI)
    schema.image_filename = schema.add_property("has filename", schema.image_class, XSD.string)
    schema.image_caption = schema.add_property("has caption", schema.image_class, XSD.string)
    schema.image_generated_caption = schema.add_property("has generated caption", schema.image_class, XSD.string)

    # Add relations
    schema.has_reference_rel=schema.add_property("has reference", concept_class, article_class)
    # Adding existing and new relationships to the schema

    # Person-Class Relationships
    schema.add_property("participant", person_class, event_class)
    schema.add_property("has nationality or citizenship", person_class, nationality_group)
    schema.add_property("lives in", person_class, gpe_class)
    schema.add_property("works at", person_class, organization_class)
    schema.add_property("associated with", person_class, facility_class)
    schema.add_property("employer", person_class, person_class)
    schema.add_property("peer", person_class, person_class)
    schema.add_property("colleague", person_class, person_class)
    schema.add_property("conflict", person_class, person_class)
    schema.add_property("couple", person_class, person_class)
    schema.add_property("family", person_class, person_class)
    schema.add_property("associated with", person_class, product_class)
    schema.add_property("user", person_class, product_class)
    schema.add_property("owner", person_class, product_class)
    schema.add_property("attended", person_class, event_class)
    schema.add_property("spokesperson for", person_class, organization_class)
    schema.add_property("visited", person_class, location_class)
    schema.add_property("represented by", person_class, organization_class)
    schema.add_property("has mentor", person_class, person_class)
    schema.add_property("mentor of", person_class, person_class)

    # Organization-Class Relationships
    schema.add_property("organized by", event_class, organization_class)
    schema.add_property("headquartered at", organization_class, gpe_class)
    schema.add_property("subsidiary of", organization_class, organization_class)
    schema.add_property("parent company of", organization_class, organization_class)
    schema.add_property("associated with", organization_class, nationality_group)
    schema.add_property("has facility", organization_class, facility_class)
    schema.add_property("employer", organization_class, person_class)
    schema.add_property("manufacturer", organization_class, product_class)
    schema.add_property("partnership", organization_class, organization_class)
    schema.add_property("competitor", organization_class, organization_class)
    schema.add_property("owns", organization_class, facility_class)
    schema.add_property("invested in", organization_class, product_class)
    schema.add_property("sponsor", organization_class, event_class)
    schema.add_property("provides service to", organization_class, person_class)
    schema.add_property("supply chain partner", organization_class, organization_class)
    schema.add_property("has branch in", organization_class, gpe_class)

    # Event-Class Relationships
    schema.add_property("held at", event_class, gpe_class)
    schema.add_property("held at", event_class, facility_class)
    schema.add_property("held at", event_class, location_class)
    schema.add_property("organized by", event_class, organization_class)
    schema.add_property("attended by", event_class, person_class)
    schema.add_property("sponsored by", event_class, organization_class)
    schema.add_property("involved in", event_class, gpe_class)
    schema.add_property("triggered by", event_class, event_class)
    schema.add_property("resulted in", event_class, event_class)

    # GPE and Location Relationships
    schema.add_property("located at", gpe_class, location_class)
    schema.add_property("located near", gpe_class, gpe_class)
    schema.add_property("allies", gpe_class, gpe_class)
    schema.add_property("conflict", gpe_class, gpe_class)
    schema.add_property("has facility", gpe_class, facility_class)
    schema.add_property("associated with", gpe_class, product_class)
    schema.add_property("manufactured at", gpe_class, product_class)
    schema.add_property("bordering", gpe_class, gpe_class)
    schema.add_property("export partner", gpe_class, gpe_class)
    schema.add_property("import partner", gpe_class, gpe_class)
    schema.add_property("diplomatic relations with", gpe_class, gpe_class)
    schema.add_property("part of", gpe_class, organization_class)

    # Product-Class Relationships
    schema.add_property("manufactured at", product_class, facility_class)
    schema.add_property("used by", product_class, person_class)
    schema.add_property("sold by", product_class, organization_class)
    schema.add_property("distributed by", product_class, organization_class)
    schema.add_property("licensed by", product_class, organization_class)
    schema.add_property("related to", product_class, product_class)

    # Facility-Class Relationships
    schema.add_property("located at", facility_class, location_class)
    schema.add_property("operated by", facility_class, organization_class)
    schema.add_property("constructed by", facility_class, organization_class)
    schema.add_property("visited by", facility_class, person_class)
    schema.add_property("hosts", facility_class, event_class)

    schema.has_depiction = schema.add_property("has depiction", concept_class, schema.depiction_class)
    schema.has_image=schema.add_property("has image", schema.depiction_class,schema.image_class)
    schema.bounding_box=schema.add_property("has bounding box", schema.depiction_class, XSD.string)

    return schema
from fastapi import FastAPI
from pydantic import BaseModel
from Tools.Tool import Squall, SparqlTool, WikiTool
from contextlib import asynccontextmanager
from refined.inference.processor import Refined
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]


def load_refined_model():
    refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                      entity_set="wikipedia")
    return refined

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["refined"] = load_refined_model
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

class Universal_Request_Body(BaseModel):
    action: str
    action_input: str


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def root():
    return {"message":"base route"}

@app.post("/fetch_observation")
async def get_items(body:Universal_Request_Body):
    if body.action == "GenerateSquall":
        refined = ml_models["refined"]()
        squall = Squall("/home/work/Human_IQ/HumanIQ_Tool/app/Tools/Tools_Data/squall_fixed_few_shot.json",refined)
        response = squall.generate_squall_query(body.action_input)
        return {"message": response}
    # elif body.action == "squall2sparql":
    #     converter = SparqlTool("/home/dhananjay/HybridQA/Tools/Tools_Data/squall2sparql_revised.sh")
    #     response = converter.gen_sparql_from_squall(body.action_input)
    #     return {"message": response}
    elif body.action == "WikiSearch":
        response = WikiTool().get_wikipedia_summary_keyword(body.action_input)
        return {"message": response}
    elif body.action == "WikiSearchSummary":
        paragraphs= WikiTool().get_wikipedia_summary(body.action_input)
        return {"message":paragraphs}
    elif body.action == "GetWikidataID":
        response = WikiTool().all_wikidata_ids(body.action_input)
        return {"message": response}
    elif body.action == "RunSparql":
        converter = SparqlTool("/home/work/Human_IQ/HumanIQ_Tool/app/Tools/Tools_Data/squall2sparql_revised.sh")
        response = converter.run_sparql(body.action_input)
        json_compatible_item_data = jsonable_encoder(response)
        return JSONResponse(content=json_compatible_item_data)
    elif body.action == "GetLabel":
        response = WikiTool().get_label(body.action_input)
        return {"message": response}
    else:
        return {"message":"It is working Test"}



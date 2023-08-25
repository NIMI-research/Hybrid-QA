import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from refined.inference.processor import Refined

from Tools.Tool import Squall, SparqlTool, WikiTool

LOGGER = logging.getLogger(__name__)

origins = ["*"]


def load_refined_model():
    refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                      entity_set="wikipedia")
    return refined


ml_models = {}


#
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["refined"] = load_refined_model
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Universal_Request_Body(BaseModel):
    action: str
    action_input: str


@app.post('/generate_squall')
async def get_items(body: Universal_Request_Body):
    try:
        refined = ml_models["refined"]()
        squall = Squall("Tools/Tools_Data/squall_fixed_few_shot.json",
                        refined)
        response = squall.generate_squall_query(body.action_input)
        return {"message": response}
    except Exception as ex_err:
        print(str(ex_err))
    # elif body.action == "squall2sparql":
    #     converter = SparqlTool("/home/dhananjay/HybridQA/Tools/Tools_Data/squall2sparql_revised.sh")
    #     response = converter.gen_sparql_from_squall(body.action_input)
    #     return {"message": response}


@app.post('/wiki_search')
async def get_wiki_search(body: Universal_Request_Body):
    try:
        response = WikiTool().get_wikipedia_summary_keyword(body.action_input)
        return {"message": response}
    except Exception as ex_Err:
        raise HTTPException(400, detail="failed wiki search")


@app.post('/wiki_search_summary')
async def get_wiki_search(body: Universal_Request_Body):
    try:
        paragraphs = WikiTool().get_wikipedia_summary(body.action_input)
        return {"message": paragraphs}
    except Exception as ex_err:
        raise HTTPException(400, detail="Failed to search wiki summary")


@app.post('/get_wikidata_id')
async def get_wiki_search(body: Universal_Request_Body):
    try:
        response = WikiTool().all_wikidata_ids(body.action_input)
        return {"message": response}
    except Exception as ex_Err:
        raise HTTPException(400, detail="Failed ")


@app.post('/run_sparql')
async def get_wiki_search(body: Universal_Request_Body):
    try:
        converter = SparqlTool("Tools/Tools_Data/squall2sparql_revised.sh")
        response = converter.run_sparql(body.action_input)
        json_compatible_item_data = jsonable_encoder(response)
        return JSONResponse(content=json_compatible_item_data)
    except Exception as ex_err:
        raise HTTPException(400, detail="Failed to run sqarql")


@app.post('/get_label')
async def get_wiki_search(body: Universal_Request_Body):
    try:
        response = WikiTool().get_label(body.action_input)
        return {"message": response}
    except Exception as ex_err:
        raise HTTPException(400, detail="Failed to get label")

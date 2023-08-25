"""
observation creation
"""
import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, HTTPException
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from refined.inference.processor import Refined

from app.Tools.Tool import Squall, SparqlTool, WikiTool

LOGGER = logging.getLogger(__name__)


def load_refined_model():
    refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                      entity_set="wikipedia")
    return refined


# ml_models = {}
#
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Load the ML model
#     ml_models["refined"] = load_refined_model
#     yield
#     # Clean up the ML models and release the resources
#     ml_models.clear()


class Universal_Request_Body(BaseModel):
    action: str
    action_input: str


# noinspection PyUnusedLocal
class ObservationRouter:
    """
    observation model api
    """

    @property
    def router(self):
        """
        observation apis
        """
        api_router = APIRouter(prefix='/api/v1/fetch_observation', tags=["Observation"])

        @api_router.post('/generate_squall')
        async def get_items(body: Universal_Request_Body):
            try:
                import pdb;pdb.set_trace()
                refined = ml_models["refined"]()
                squall = Squall("app/Tools/Tools_Data/squall_fixed_few_shot.json",
                                refined)
                response = squall.generate_squall_query(body.action_input)
                return {"message": response}
            except Exception as ex_err:
                raise HTTPException(400, detail="Failed general search")
            # elif body.action == "squall2sparql":
            #     converter = SparqlTool("/home/dhananjay/HybridQA/Tools/Tools_Data/squall2sparql_revised.sh")
            #     response = converter.gen_sparql_from_squall(body.action_input)
            #     return {"message": response}

        @api_router.post('/wiki_search')
        async def get_wiki_search(body: Universal_Request_Body):
            try:
                response = WikiTool().get_wikipedia_summary_keyword(body.action_input)
                return {"message": response}
            except Exception as ex_Err:
                raise HTTPException(400, detail="failed wiki search")

        @api_router.post('/wiki_search_summary')
        async def get_wiki_search(body: Universal_Request_Body):
            try:
                paragraphs = WikiTool().get_wikipedia_summary(body.action_input)
                return {"message": paragraphs}
            except Exception as ex_err:
                raise HTTPException(400, detail="Failed to search wiki summary")

        @api_router.post('/get_wikidata_id')
        async def get_wiki_search(body: Universal_Request_Body):
            try:
                response = WikiTool().all_wikidata_ids(body.action_input)
                return {"message": response}
            except Exception as ex_Err:
                raise HTTPException(400, detail="Failed ")

        @api_router.post('/run_sparql')
        async def get_wiki_search(body: Universal_Request_Body):
            try:
                converter = SparqlTool("app/Tools/Tools_Data/squall2sparql_revised.sh")
                response = converter.run_sparql(body.action_input)
                json_compatible_item_data = jsonable_encoder(response)
                return JSONResponse(content=json_compatible_item_data)
            except Exception as ex_err:
                raise HTTPException(400, detail="Failed to run sqarql")

        @api_router.post('/get_label')
        async def get_wiki_search(body: Universal_Request_Body):
            try:
                response = WikiTool().get_label(body.action_input)
                return {"message": response}
            except Exception as ex_err:
                raise HTTPException(400, detail="Failed to get label")

        return api_router

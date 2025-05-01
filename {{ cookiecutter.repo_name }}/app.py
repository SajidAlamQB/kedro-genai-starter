from collections import defaultdict
from contextlib import asynccontextmanager
from importlib.metadata import version

import structlog
from fastapi import FastAPI, HTTPException
from kedro.framework.cli.utils import ENTRY_POINT_GROUPS, _get_entry_points
from kedro.framework.session import KedroSession
from kedro.framework.startup import configure_project
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    package_name: str = "{{ cookiecutter.python_package }}"
    kedro_env: str = "local"

    model_config = SettingsConfigDict(env_file=".env")


logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI to configure Kedro project.
    """
    logger.info("Configuring Kedro project...")
    configure_project(settings.package_name)
    logger.info("Kedro project configured")

    yield


settings = Settings()
app = FastAPI(lifespan=lifespan)


def get_kedro_info():
    # Inspired by https://github.com/kedro-org/kedro/blob/0.19.12/kedro/framework/cli/cli.py
    plugin_versions = {}
    plugin_entry_points = defaultdict(set)
    for plugin_entry_point in ENTRY_POINT_GROUPS:
        for entry_point in _get_entry_points(plugin_entry_point):
            module_name = entry_point.module.split(".")[0]
            plugin_versions[module_name] = entry_point.dist.version
            plugin_entry_points[module_name].add(plugin_entry_point)

    return {
        "kedro_version": version("kedro"),
        "plugins": plugin_versions,
        "entry_points": plugin_entry_points,
    }


@app.get("/")
async def home():
    return get_kedro_info()


@app.get("/registry")
async def registry():
    """
    Returns the pipeline registry of the Kedro project.
    """
    from kedro.framework.project import pipelines as pipeline_mapping

    return {
        "pipelines": [
            {
                "name": pipeline_name,
                "nodes": [
                    {
                        "name": node.name,
                        "inputs": node.inputs,
                        "outputs": node.outputs,
                    }
                    for node in pipeline.nodes
                ],
            }
            for pipeline_name, pipeline in pipeline_mapping.items()
        ]
    }


@app.post("/run/{pipeline_name}")
async def run_pipeline(pipeline_name: str):
    """
    Runs a Kedro pipeline and returns the result.
    """
    with KedroSession.create(env=settings.kedro_env) as session:
        try:
            output = session.run(pipeline_name=pipeline_name)
        except Exception as e:
            raise HTTPException(
                500,
                detail={
                    "success": False,
                    "error": str(e),
                },
            )

    logger.info(
        "Pipeline run completed",
        pipeline_name=pipeline_name,
        output_vars=list(output.keys()),
    )
    return {
        "success": True,
        "output_vars": list(output.keys()),
    }

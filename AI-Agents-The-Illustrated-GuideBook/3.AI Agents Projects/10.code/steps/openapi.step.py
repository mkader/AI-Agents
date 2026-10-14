from api_models import RequestBody, SuccessResponse, ErrorResponse


config = {
    'type': 'api',
    'name': 'OpenAPI',
    'description': 'OpenAPI document for content generation',
    'path': '/openapi.json',
    'method': 'GET',
    'emits': []
}


async def handler(req, context):
    return {
        'status': 200,
        'body': {
            'openapi': '3.1.0',
            'info': {'title': 'VibeStudio API', 'version': '1.0.0'},
            'servers': [{'url': '/'}],
            'paths': {
                '/generate-content': {
                    'post': {
                        'operationId': 'generateContent',
                        'summary': 'Start content generation from an article URL',
                        'description': 'Returns a trace ID immediately. Scraping, generation and draft scheduling run asynchronously; inspect the Motia workbench logs for completion or failures.',
                        'requestBody': {
                            'required': True,
                            'content': {
                                'application/json': {
                                    'schema': RequestBody.model_json_schema(),
                                    'example': {'url': 'https://example.com/article'}
                                }
                            }
                        },
                        'responses': {
                            '200': {
                                'description': 'Workflow started (not yet completed)',
                                'content': {'application/json': {'schema': SuccessResponse.model_json_schema()}}
                            },
                            '400': {
                                'description': 'Invalid request body',
                                'content': {'application/json': {'schema': ErrorResponse.model_json_schema()}}
                            },
                            '500': {
                                'description': 'Internal server error',
                                'content': {'application/json': {'schema': ErrorResponse.model_json_schema()}}
                            }
                        }
                    }
                }
            }
        }
    }

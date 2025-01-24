import json
import logging
import sys
import traceback

from django.http import JsonResponse
from rest_framework.exceptions import NotAuthenticated, PermissionDenied
from rest_framework.views import exception_handler

log = logging.getLogger(__name__)


def ExceptionHandler(exc, context):
    response = exception_handler(exc, context)

    if 'test' not in sys.argv and type(exc) not in [
        NotAuthenticated,
        PermissionDenied,
    ]:
        response = JsonResponse(
            {'status_code': 500, 'message': 'an unknown error occured'}
        )
        response.status_code = 500
        log.error(json.dumps({
            'event': 'UncaughtException',
            'traceback': traceback.format_exc()
        }))
    return response

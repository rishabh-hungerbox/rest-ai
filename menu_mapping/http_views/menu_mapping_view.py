from django.http import HttpResponse, JsonResponse
from rest_framework.views import APIView


class MenuMapperAIView(APIView):
    """ Get a particular Bill in HTML or Json Format"""
    def get(self, request, bill_id):
        return JsonResponse({'message': "hi"})

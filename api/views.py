from pprint import pprint
from rest_framework.request import Request

from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse, HttpResponse
from rest_framework.decorators import api_view
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from api import services


@api_view(['POST'])
def detect_topics(request: Request):
    print(request.data.keys())
    if not ("input" in request.data.keys()):
        return JsonResponse({'message': 'body with key input is missed'}, status=status.HTTP_400_BAD_REQUEST,
                            safe=False)
    api_service = services.ApiService()
    return JsonResponse(api_service.getTopicDetectionResult(input=request.data['input']),
                        status=status.HTTP_200_OK, safe=False)


@api_view(['POST'])
def detect_topics_bertopic(request: Request):
    if not ("input" in request.data.keys()):
        return JsonResponse({'message': 'body with key input is missed'}, status=status.HTTP_400_BAD_REQUEST,
                            safe=False)
    api_service = services.ApiService()
    return JsonResponse(api_service.get_topic_detection_by_bertopic(input=request.data['input']),
                        status=status.HTTP_200_OK, safe=False)

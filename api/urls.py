from django.urls import path
from .views import query_llm, hello_world

urlpatterns = [
    path('query/', query_llm, name='query_llm'),
    path('hello/', hello_world, name='hello_world'),
]

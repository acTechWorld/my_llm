from django.urls import path
from .views import query_llm

urlpatterns = [
    path('query/', query_llm, name='query_llm'),
]

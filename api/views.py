from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .llm import generate_response
import json

@csrf_exempt
def query_llm(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            prompt = data.get('prompt', '')
            if not prompt:
                return JsonResponse({'error': 'Prompt is required'}, status=400)

            # Get response from LLM
            response = generate_response(prompt)
            return JsonResponse({'response': response})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

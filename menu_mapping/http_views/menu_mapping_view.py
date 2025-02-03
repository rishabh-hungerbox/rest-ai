from django.http import HttpResponse, JsonResponse
from rest_framework.views import APIView
from menu_mapping.helper_classes.menu_mapper_helper import get_master_menu_response, process_data
from menu_mapping.helper_classes.utility import MenuMappingUtility
from io import TextIOWrapper
import csv
import threading


class MenuMapperAIView(APIView):
    """ Get relevant master menus based on provided child menu name"""
    def get(self, request):
        menu_name = request.query_params.get('menu_name')
        relevant_items = get_master_menu_response(menu_name)
        return JsonResponse({'result': relevant_items})
    
    def post(self, request):
        file = request.FILES['file']
        
        if not file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        
        if not file.name.endswith('.csv'):
            return JsonResponse({'error': 'File is not a CSV'}, status=400)
        
        try:
            decoded_file = TextIOWrapper(file.file, encoding='utf-8')
            reader = csv.DictReader(decoded_file)
            expected_headers = {'id', 'name', 'order_count', 'qty', 'mv_id', 'mv_name', 'description'}
            
            if set(reader.fieldnames) != expected_headers:
                return JsonResponse({'error': 'CSV headers do not match the expected format'}, status=400)
            
            def process_rows():
                input_data = {}
                rows = list(reader)
                sorted_rows = sorted(rows, key=lambda row: int(row['id']))
                for row in sorted_rows:
                    normalized_item_name = MenuMappingUtility.normalize_string(row['name'])
                    input_data[row['id']] = {
                        "id": row['id'],
                        "name": normalized_item_name,
                        "mv_id": row['mv_id'],
                        "mv_name": row['mv_name']
                    }
                process_data(input_data)
            
            thread = threading.Thread(target=process_rows)
            thread.start()
            
            return JsonResponse({'message': 'File sent for Processing!'})
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        

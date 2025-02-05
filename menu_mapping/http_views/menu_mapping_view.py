from django.http import HttpResponse, JsonResponse
from rest_framework.views import APIView
from menu_mapping.helper_classes.menu_mapper_helper import get_master_menu_response, process_data
from menu_mapping.helper_classes.utility import MenuMappingUtility
from io import TextIOWrapper
import csv
from menu_mapping.models import MenuMappingPrediction
import threading


class MenuMapperAIView(APIView):
    """ Get relevant master menus based on provided child menu name"""
    def get(self, request):
        menu_name = request.query_params.get('menu_name')
        relevant_items = get_master_menu_response(menu_name)
        return JsonResponse({'result': relevant_items})
    
    def post(self, request):
        file = request.FILES['file']
        log_id = request.data.get('log_id')
        
        if not file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        
        if not file.name.endswith('.csv'):
            return JsonResponse({'error': 'File is not a CSV'}, status=400)
        
        try:
            decoded_file = TextIOWrapper(file.file, encoding='utf-8')
            reader = csv.DictReader(decoded_file)
            expected_headers = {'id', 'name', 'order_count', 'mv_id', 'mv_name', 'description'}
            
            if set(reader.fieldnames) != expected_headers:
                return JsonResponse({'error': 'CSV headers do not match the expected format'}, status=400)
            
            def process_rows(log_id):
                done_menu_ids = {}
                if log_id:
                    done_menu_ids = MenuMappingPrediction.objects.filter(log_id=log_id).values_list('menu_id', flat=True)
                input_data = {}
                rows = list(reader)
                sorted_rows = sorted(rows, key=lambda row: int(row['id']))
                for row in sorted_rows:
                    if int(row['id']) in done_menu_ids:
                        print(f'skip {row['id']}')
                        continue
                    normalized_item_name = MenuMappingUtility.normalize_string(row['name'])
                    print(f'add {row['id']}')
                    input_data[row['id']] = {
                        "id": row['id'],
                        "name": normalized_item_name,
                        "mv_id": row['mv_id'],
                        "mv_name": row['mv_name']
                    }
                process_data(input_data, log_id)
            
            thread = threading.Thread(target=process_rows, args=[log_id])
            thread.start()
            
            return JsonResponse({'message': 'File sent for Processing!'})
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        

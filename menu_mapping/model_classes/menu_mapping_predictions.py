from django.db import models
from menu_mapping.models import LLMLogs


class MenuMappingPrediction(models.Model):
    id = models.AutoField(primary_key=True)
    menu_id = models.IntegerField()
    menu_name = models.CharField(max_length=255)
    corrected_menu_name = models.CharField(max_length=255)
    quantitative_menu_name = models.CharField(max_length=255, null=True)
    master_menu_id = models.IntegerField()
    master_menu_name = models.CharField(max_length=255)
    eval_current = models.BooleanField(default=False)
    predicted_menu_name = models.CharField(max_length=255)
    eval_prediction = models.BooleanField(default=None, null=True)
    log = models.ForeignKey(LLMLogs, on_delete=models.DO_NOTHING)  # Foreign key to LLMLogs
    response = models.TextField()
    ranked_nodes = models.JSONField()
    is_approved = models.BooleanField(null=True, default=None)

    class Meta:
        db_table = 'menu_mapping_predictions'

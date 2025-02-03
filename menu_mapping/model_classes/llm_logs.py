from django.db import models


class LLMLogs(models.Model):
    id = models.AutoField(primary_key=True)
    model_name = models.CharField(max_length=255)
    embedding_model = models.CharField(max_length=255)
    prompt = models.TextField()

    class Meta:
        db_table = 'llm_logs'

# Generated by Django 4.2.1 on 2024-01-23 11:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app", "0003_activeuser"),
    ]

    operations = [
        migrations.CreateModel(
            name="UserReport",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=255)),
                ("message", models.TextField()),
                ("timestamp", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]

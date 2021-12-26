from django import forms

class UserForm(forms.Form):
    first_name= forms.CharField(max_length=100)

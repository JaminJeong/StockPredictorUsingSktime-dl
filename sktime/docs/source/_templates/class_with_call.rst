:mod:`{{module}}`.{{objname}}
{{ underline }}===============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__
   .. automethod:: __call__
   {% endblock %}

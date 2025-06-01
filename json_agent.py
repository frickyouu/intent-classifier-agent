import json
from typing import Dict, Optional, Union
from pydantic import BaseModel, ValidationError


class TargetSchema(BaseModel):
    """Target schema for JSON transformation"""
    id: str
    name: str
    timestamp: str
    priority: Optional[str] = "normal"
    metadata: Dict[str, str]


class JSONAgent:
    def __init__(self, schema_validator: BaseModel = TargetSchema):
        self.schema_validator = schema_validator

    def process(self, json_payload: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Processes a JSON payload and validates against the schema.

        Args:
            json_payload: Raw JSON string or dict

        Returns:
            Dict containing:
            - 'data': Validated and normalized payload (if successful)
            - 'errors': List of validation errors
            - 'status': 'success' or 'error'
        """
        result = {
            'data': None,
            'errors': [],
            'status': 'success',
            'metadata': metadata or {} 

        }

        try:
            # Accept either JSON string or dict
            input_data = (
                json.loads(json_payload)
                if isinstance(json_payload, str)
                else json_payload
            )

            # Validate using the schema
            validated_data = self.schema_validator(**input_data)
            result['data'] = validated_data.dict()

        except json.JSONDecodeError as e:
            result['status'] = 'error'
            result['errors'].append(f"Invalid JSON format: {str(e)}")

        except ValidationError as e:
            result['status'] = 'error'
            for error in e.errors():
                field = error['loc'][0]
                msg = error['msg']
                result['errors'].append(f"Field '{field}': {msg}")

        return result


json_format = \
{
    "type": "function",
    "function": 
    {
        "name": "menu_analysis",
        "description": "Anlysis menu data from texts",
        "parameters": 
        {
            "type": "object",
            "properties": 
            {
                "restaurant_name": 
                {
                    "type": "string",
                    "description": "restaurant name from texts"
                },
                "business_hours": 
                {
                    "type": "string",
                    "description": "business hours from texts"
                },
                "contact": 
                {
                    "type": "object",
                    "properties": 
                    {
                        "address": 
                        {
                            "type": "string",
                            "description": "restaurant address from texts"
                        },
                        "phone": 
                        {
                            "type": "string",
                            "description": "restaurant phone number from texts"
                        }
                    },
                },
                "dish": 
                {
                    "type": "array",
                    "description": "dish info form texts, you have to analysis all of results",
                    "items": 
                    {
                        "type": "object",
                        "properties": 
                        {
                            "name": 
                            {
                                "type": "string",
                                "description": "dish name from texts"
                            },
                            "price": 
                            {
                                "type": "string",
                                "description": "dish price from texts"
                            },
                        },
                    }
                },
            },
            "required": ["dish"]
        },
    }
}

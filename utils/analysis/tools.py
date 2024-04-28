TOOL_1 = {
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
                    }
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
                            }
                        }
                    }
                },
                "other": 
                {
                    "type": "string",
                    "description": "other info except restaurant_name, business_hours, contact, dish"
                }
            },
            "required": ["dish", "name", "price"]
        }
    }
}

TOOL_2 = {
    "type": "function",
    "function": 
    {
        "name": "menu_analysis",
        "description": "given menu text json data which have bounding polynimal vertex data from OCR",
        "parameters":
        {
            "type": "object",
            "properties":
            {
                "restaurant_name": 
                {
                    "type": "string",
                    "description": "determine the restaurant name based on bounding polynimal vertex and text"
                },
                "business_hours": 
                {
                    "type": "string",
                    "description": "determine the business hours based on bounding polynimal vertex and text"
                },
                "contact": 
                {
                    "type": "object",
                    "properties": 
                    {
                        "address": 
                        {
                            "type": "string",
                            "description": "determine the restaurant address based on bounding polynimal vertex and text"
                        },
                        "phone": 
                        {
                            "type": "string",
                            "description": "determine the restaurant phone number based on bounding polynimal vertex and text"
                        }
                    }
                },
                "dish": 
                {
                    "type": "array",
                    "description": "determine the dish information based on bounding polynimal vertex and text, you have to analysis all of results",
                    "items": 
                    {
                        "type": "object",
                        "properties": 
                        {
                            "name": 
                            {
                                "type": "string",
                                "description": "determine the dish name based on bounding polynimal vertex and text"
                            },
                            "price": 
                            {
                                "type": "string",
                                "description": "determine the dish price based on bounding polynimal vertex and text"
                            }
                        }
                    }
                },
                "other": 
                {
                    "type": "string",
                    "description": "other info except restaurant_name, business_hours, contact, dish"
                }
            },
            "required": ["dish"]
        }
    }
}
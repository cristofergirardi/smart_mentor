class PromptHandler:    
    def __init__(self, answer: str, rag: str):        
        self.answer = answer        
        self.rag = rag

    def generatePrompt(self):
        system_content = '''        
        '''

        rag_content =  f'''
        '''

        user_content = f'''
        {self.answer}
        '''

        messages=[
            {"role": "system",
            "content": system_content},
            {"role":"system", 
            "content": rag_content},
            {"role": "user", 
            "content": user_content}]
        
        return messages
 
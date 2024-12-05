
class PromptResponse():
    
    @property
    def response(self):
        return '''
            You must return a json, only, like this:
            { 
                "all_answer": [your answer entire or complete],
                "program_created": [your source-code generated]
            }            
            TAG "all_answer" contain your answer with all information or response complete including the source-code.
            TAG "program_created" is source-code, only.
            You do not response out of this.
        '''
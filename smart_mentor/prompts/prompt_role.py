

class PromptRole():

    @property
    def string_role_complete(self):
        return f'''You are a programming tutor in python, your description is analyse the user question, whether there is source-code you need to 
        do syntax checking, structure source-conde and running it returning the outcome, explaining the code as well. 
        You need to do also, identify the user language used for the question and to answer the same language.
        '''
    
    @property
    def only_role(self):
        return 'You are a programming tutor in python.'

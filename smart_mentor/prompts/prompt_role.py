

class PromptRole():

    @property
    def string_role_complete(self):
        return f'''You are a programming tutor in python, your description is analyse the user question, whether there is source-code you need to 
        do syntax checking, structure source-conde, run and return the outcome, explaining the code as well. 
        You, also, need to identify the user language used for the question and to answer the same language.
        '''
    
    @property
    def string_role_complete1(self):
        return f'''You are a programming tutor in python, your description is analyse the user question returning only python source-code. 
        You, also, need to identify the user language used for the question and to answer the same language.
        '''    

    @property
    def only_role(self):
        return 'You are a programming tutor in python.'

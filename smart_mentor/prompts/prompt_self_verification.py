class PromptSelfVerification():
    
    @property
    def self_verification(self):
        return f'''
            Please rewritte the question and answer to give better response.
        '''
def calendarioVacinas(idade: str) -> str:
    """
    Exibe uma lista de vacinas disponíveis baseadas na idade do paciente.
    Não retorna preço se não houver estoque disponível. Informa sempre o valor por dose.
    Se a vacina pesquisada não retornar resultado, oriente o cliente a conversar com um consultor.

    Args:
        idade: A idade do paciente (ex: '2 meses', '1 ano', 'Adulto', '60 anos').

    Returns:
        Um texto simulando o calendário de vacinas recomendadas, com estoque e valor por dose.
    """
    idade = idade.lower()
    
    # Simulação de um banco de dados para a área comercial.  
    # Na implementação real, isso seria uma consulta SQL/API.
    
    # Exemplo: Se for bebê
    if "mes" in idade or "mês" in idade or "meses" in idade:
        return """
        Resultado da busca por idade (Bebês):
        1. Vacina: Hexavalente (Protege contra Difteria, Tétano, Coqueluche, Haemophilus tipo b, Hepatite B e Poliomielite)
           Estoque: Disponível
           Valor da dose: R$ 380,00
           Esquema: 2, 4 e 6 meses (com reforço aos 15 meses)
           
        2. Vacina: Rotavírus Pentavalente
           Estoque: Indisponível no momento
           Valor da dose: (Sem preço, informar indisponibilidade)
           Esquema: 2, 4 e 6 meses
           
        3. Vacina: Pneumocócica 13 ou 15 Valente
           Estoque: Disponível
           Valor da dose: R$ 350,00
           Esquema: 2, 4 e 6 meses (reforço entre 12 e 15 meses)
        """
        
    # Exemplo: Adulto/Qualquer outra vacina comum de interesse
    if "adulto" in idade or "anos" in idade:
        return """
        Resultado da busca (Adultos):
        1. Vacina: Qdenga (dengue)
           Estoque: Disponível
           Valor da dose: R$ 450,00
           Esquema: 2 doses (intervalo de 3 meses)

        2. Vacina: HPV Nonalvalente
           Estoque: 0
           Valor da dose: (Sem preço, pois não há estoque)
           Esquema: 2 ou 3 doses dependendo da idade

        3. Vacina: Gripe Quadrivalente (Influenza)
           Estoque: Disponível
           Valor da dose: R$ 120,00
           Esquema: Anual
        """
        
    return "Nenhuma vacina encontrada ou idade não especificada de forma clara. Sugira que um responsável técnico fará o atendimento."

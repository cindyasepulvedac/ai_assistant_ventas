
from azure.cosmos import CosmosClient
from azure.identity import ClientSecretCredential, DefaultAzureCredential, ManagedIdentityCredential
from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError, HttpResponseError
from azure.keyvault.secrets import SecretClient
import logging


def get_cosmos_client(client_id, client_secret, tenant_id, vault_uri, comosdb_key_secret_name, comosdb_uri_secret_name):
    """
    Obtiene un cliente de CosmosDB para interactuar con una cuenta de Cosmos DB, recuperando las credenciales
    desde Azure Key Vault usando un Service Principal.

    Args:
        client_id (str): ID del cliente (aplicación) registrado en Azure Active Directory.
        client_secret (str): Secreto del cliente para autenticación del Service Principal.
        tenant_id (str): ID del tenant de Azure Active Directory.
        vault_uri (str): URI completo del Azure Key Vault (ejemplo: "https://mykeyvault.vault.azure.net/").
        comosdb_key_secret_name (str): Nombre del secreto en Key Vault que contiene la clave de la cuenta Cosmos DB.
        comosdb_uri_secret_name (str): Nombre del secreto en Key Vault que contiene el URI de la cuenta Cosmos DB.
        use_managed_identity (bool): Si True, usa la identidad administrada del servicio en lugar 
                                   de Service Principal.

    Returns:
        CosmosClient: Cliente de Cosmos DB inicializado con las credenciales recuperadas.
    """
    credential = ClientSecretCredential(
                client_id=client_id,
                client_secret=client_secret,
                tenant_id=tenant_id
            )     
    try:
        secret_client = SecretClient(vault_url=vault_uri, credential=credential)
        
        try:
            cosmosdb_key = secret_client.get_secret(comosdb_key_secret_name).value
            cosmosdb_uri = secret_client.get_secret(comosdb_uri_secret_name).value
        except ResourceNotFoundError as e:
            logging.error(f"Secreto no encontrado en Key Vault: {str(e)}")
            raise
            
        cosmos_client = CosmosClient(url=cosmosdb_uri, credential=cosmosdb_key)
        return cosmos_client
        
    except ClientAuthenticationError as e:
        logging.error(f"Error de autenticación: {str(e)}")
        raise
    except HttpResponseError as e:
        logging.error(f"Error al conectar con Azure: {str(e)}")
        raise

def insert_dict_to_cosmosdb(cosmos_client, database_name, container_name, new_item):
    """
    Inserta o actualiza un documento en CosmosDB a partir de un diccionario.

   Args:
       cosmos_client (CosmosClient): Cliente inicializado de Cosmos DB.
       database_name (str): Nombre de la base de datos en Cosmos DB.
       container_name (str): Nombre del contenedor donde se insertará el documento.
       new_item (dict): Diccionario con los datos a insertar. Debe contener un campo 'id' 
                       si se desea actualizar un documento existente.

   Returns:
       dict: Documento creado/actualizado con metadata de Cosmos DB.

    """
    if not isinstance(new_item, dict):
        raise ValueError("new_item debe ser un diccionario")

    database = cosmos_client.get_database_client(database_name)
    container = database.get_container_client(container_name)
    created_item = container.upsert_item(new_item)
    logging.info(
            f"Documento insertado/actualizado exitosamente en {database}/{container}"
            f" con id: {created_item.get('id')}"
        )

    return created_item
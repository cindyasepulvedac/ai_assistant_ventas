from azure.identity import ClientSecretCredential, DefaultAzureCredential, ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContentSettings, ContainerClient
from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError, HttpResponseError
import io
import logging

def get_container_client(client_id, client_secret, tenant_id, account_url, container_name):
    """
    Obtiene un cliente de contenedor para interactuar con un contenedor específico en Azure Blob Storage.

    Este método establece una conexión segura utilizando una credencial basada en el cliente secreto y devuelve un cliente de contenedor, 
    permitiendo realizar operaciones sobre blobs dentro del contenedor especificado. Además, valida la conexión listando los contenedores existentes.

    Args:
        client_id (str): ID de la aplicación registrada en Azure AD.
        client_secret (str): Secreto asociado a la aplicación registrada.
        tenant_id (str): ID del inquilino (tenant) de Azure Active Directory.
        account_url (str): URL de la cuenta de Azure Blob Storage (por ejemplo, "https://<nombre_cuenta>.blob.core.windows.net").
        container_name (str): Nombre del contenedor dentro de Azure Blob Storage donde se desea operar.

    Returns:
        ContainerClient: Un cliente de contenedor que permite realizar operaciones sobre el contenedor especificado.

    """  
    credential = ClientSecretCredential(
                client_id=client_id,
                client_secret=client_secret,
                tenant_id=tenant_id
            )
    try:
        blob_service_client = BlobServiceClient(account_url=account_url, credential=credential) 
        
        # Prueba de conectividad
        containers = blob_service_client.list_containers()
        logging.info("Conectado exitosamente al Storage Account")
        
        container_client = blob_service_client.get_container_client(container_name)
    
    except Exception as e:
        logging.error(f"Error al obtener container client: {str(e)}")
        raise

    return container_client

def upload_img_to_adls(container_client, blob_name, image):
    """
    Almacena una imagen en Azure Data Lake Storage (ADLS).

    Este método toma una imagen en formato PIL, la convierte en un flujo de bytes y la sube al contenedor especificado en ADLS. 
    Se utiliza el tipo de contenido adecuado para garantizar que la imagen sea reconocida correctamente como un archivo PNG.

    Args:
        container_client (ContainerClient): Cliente del contenedor donde se almacenará la imagen.
        blob_name (str): Nombre del blob bajo el cual se almacenará la imagen en el contenedor.
        image (PIL.Image.Image): Objeto de imagen en formato PIL que se desea almacenar.

    Returns:
        None

    """
    logging.info(f"Creando buffer para la imagen...")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_byte_arr.seek(0) 
        
    blob_client = container_client.get_blob_client(blob_name)
    image_content_setting = ContentSettings(content_type='image/png')
    blob_client.upload_blob(img_byte_arr,overwrite=True,content_settings=image_content_setting)
    
    logging.info(f"Imagen almacenada exitosamente en ADLS bajo el nombre {blob_name}")

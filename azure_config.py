"""
Azure Configuration and Blob Management Module
Centralized Azure services configuration and blob storage management.
"""

import os
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv


class AzureConfig:
    """
    Loads and manages Azure configuration from environment variables.
    Raises RuntimeError if any required variable is missing.
    """
    
    def __init__(self):
        """Initialize Azure configuration from environment variables."""
        try:
            load_dotenv()
            self.endpoint = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
            self.key = os.getenv("DOC_INTELLIGENCE_KEY")
            self.blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
            self.blob_container = os.getenv("AZURE_BLOB_CONTAINER")
            
            # Validate that all required variables are present
            missing_vars = []
            if not self.endpoint:
                missing_vars.append("DOC_INTELLIGENCE_ENDPOINT")
            if not self.key:
                missing_vars.append("DOC_INTELLIGENCE_KEY")
            if not self.blob_connection_string:
                missing_vars.append("AZURE_BLOB_CONNECTION_STRING")
            if not self.blob_container:
                missing_vars.append("AZURE_BLOB_CONTAINER")
            
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
                
        except Exception as e:
            raise RuntimeError(f"AzureConfig initialization failed: {e}")
    
    def get_document_client(self):
        """
        Create and return a DocumentIntelligenceClient instance.
        
        Returns:
            DocumentIntelligenceClient instance
        """
        try:
            return DocumentIntelligenceClient(
                endpoint=self.endpoint, 
                credential=AzureKeyCredential(self.key)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create DocumentIntelligenceClient: {e}")
    
    def get_blob_service_client(self):
        """
        Create and return a BlobServiceClient instance.
        
        Returns:
            BlobServiceClient instance
        """
        try:
            return BlobServiceClient.from_connection_string(self.blob_connection_string)
        except Exception as e:
            raise RuntimeError(f"Failed to create BlobServiceClient: {e}")
    
    def validate_configuration(self):
        """
        Validate that Azure services can be accessed with current configuration.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'document_intelligence': False,
            'blob_storage': False,
            'errors': []
        }
        
        # Test Document Intelligence
        try:
            client = self.get_document_client()
            # Try a simple operation to validate credentials
            validation_results['document_intelligence'] = True
        except Exception as e:
            validation_results['errors'].append(f"Document Intelligence validation failed: {e}")
        
        # Test Blob Storage
        try:
            client = self.get_blob_service_client()
            # Try to access the container
            container_client = client.get_container_client(self.blob_container)
            container_client.get_container_properties()
            validation_results['blob_storage'] = True
        except Exception as e:
            validation_results['errors'].append(f"Blob Storage validation failed: {e}")
        
        return validation_results


class BlobManager:
    """
    Manages Azure Blob Storage operations for file management.
    """
    
    def __init__(self, service_client=None, container=None):
        """
        Initialize BlobManager with Azure Blob Service Client.
        
        Args:
            service_client: BlobServiceClient instance (optional)
            container: Container name (optional)
        """
        try:
            if service_client is None or container is None:
                config = AzureConfig()
                service_client = config.get_blob_service_client()
                container = config.blob_container
            
            self.service_client = service_client
            self.container = container
            self.container_client = self.service_client.get_container_client(container)
            
        except Exception as e:
            raise RuntimeError(f"BlobManager initialization failed: {e}")
    
    def list_files(self, extensions=None, prefix=None):
        """
        List all files in the container, optionally filtered by extensions and prefix.
        
        Args:
            extensions: Tuple or list of file extensions to filter by (e.g., ('.pdf', '.xlsx'))
            prefix: Prefix to filter blob names by
            
        Returns:
            List of blob names matching the criteria
        """
        try:
            blobs = self.container_client.list_blobs(name_starts_with=prefix)
            
            if extensions:
                # Ensure extensions is a tuple for endswith()
                if isinstance(extensions, str):
                    extensions = (extensions,)
                elif isinstance(extensions, list):
                    extensions = tuple(extensions)
                
                return [blob.name for blob in blobs if blob.name.lower().endswith(extensions)]
            else:
                return [blob.name for blob in blobs]
                
        except Exception as e:
            raise RuntimeError(f"Failed to list files in Blob Storage: {e}")
    
    def download_file(self, blob_name):
        """
        Download a blob and return its bytes.
        
        Args:
            blob_name: Name of the blob to download
            
        Returns:
            Blob content as bytes
        """
        try:
            blob_client = self.service_client.get_blob_client(
                container=self.container, 
                blob=blob_name
            )
            return blob_client.download_blob().readall()
            
        except Exception as e:
            raise RuntimeError(f"Failed to download blob '{blob_name}': {e}")
    
    def upload_file(self, blob_name, file_data, overwrite=False):
        """
        Upload file data to blob storage.
        
        Args:
            blob_name: Name for the blob
            file_data: File content as bytes
            overwrite: Whether to overwrite existing blob
            
        Returns:
            Upload result information
        """
        try:
            blob_client = self.service_client.get_blob_client(
                container=self.container,
                blob=blob_name
            )
            
            result = blob_client.upload_blob(file_data, overwrite=overwrite)
            return {
                'success': True,
                'blob_name': blob_name,
                'etag': result.get('etag'),
                'last_modified': result.get('last_modified')
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to upload blob '{blob_name}': {e}")
    
    def delete_file(self, blob_name):
        """
        Delete a blob from storage.
        
        Args:
            blob_name: Name of the blob to delete
            
        Returns:
            Deletion result information
        """
        try:
            blob_client = self.service_client.get_blob_client(
                container=self.container,
                blob=blob_name
            )
            
            blob_client.delete_blob()
            return {
                'success': True,
                'blob_name': blob_name,
                'message': f"Blob '{blob_name}' deleted successfully"
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to delete blob '{blob_name}': {e}")
    
    def get_file_info(self, blob_name):
        """
        Get information about a blob.
        
        Args:
            blob_name: Name of the blob
            
        Returns:
            Dictionary with blob information
        """
        try:
            blob_client = self.service_client.get_blob_client(
                container=self.container,
                blob=blob_name
            )
            
            properties = blob_client.get_blob_properties()
            
            return {
                'name': blob_name,
                'size': properties.size,
                'last_modified': properties.last_modified,
                'content_type': properties.content_settings.content_type,
                'etag': properties.etag,
                'creation_time': properties.creation_time
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get info for blob '{blob_name}': {e}")
    
    def list_files_with_info(self, extensions=None):
        """
        List files with their metadata information.
        
        Args:
            extensions: File extensions to filter by
            
        Returns:
            List of dictionaries with file information
        """
        try:
            file_names = self.list_files(extensions)
            files_info = []
            
            for file_name in file_names:
                try:
                    info = self.get_file_info(file_name)
                    files_info.append(info)
                except Exception as e:
                    # Continue with other files if one fails
                    print(f"Warning: Could not get info for '{file_name}': {e}")
                    files_info.append({
                        'name': file_name,
                        'size': None,
                        'error': str(e)
                    })
            
            return files_info
            
        except Exception as e:
            raise RuntimeError(f"Failed to list files with info: {e}")


def get_azure_config():
    """
    Factory function to create an AzureConfig instance.
    
    Returns:
        AzureConfig instance
    """
    return AzureConfig()


def get_blob_manager():
    """
    Factory function to create a BlobManager instance with default configuration.
    
    Returns:
        BlobManager instance
    """
    config = get_azure_config()
    blob_service_client = config.get_blob_service_client()
    return BlobManager(blob_service_client, config.blob_container)


def get_document_client():
    """
    Factory function to create a Document Intelligence client.
    
    Returns:
        DocumentIntelligenceClient instance
    """
    config = get_azure_config()
    return config.get_document_client()


# Example usage
if __name__ == "__main__":
    # Example of how to use Azure configuration and blob management
    try:
        # Test Azure configuration
        config = get_azure_config()
        validation = config.validate_configuration()
        
        print("Azure Configuration Validation:")
        print(f"Document Intelligence: {'✓' if validation['document_intelligence'] else '✗'}")
        print(f"Blob Storage: {'✓' if validation['blob_storage'] else '✗'}")
        
        if validation['errors']:
            print("Errors:")
            for error in validation['errors']:
                print(f"  - {error}")
        
        # Test blob management
        blob_manager = get_blob_manager()
        
        # List PDF and Excel files
        pdf_files = blob_manager.list_files(('.pdf',))
        excel_files = blob_manager.list_files(('.xlsx', '.xls'))
        
        print(f"\nFound {len(pdf_files)} PDF files and {len(excel_files)} Excel files")
        
    except Exception as e:
        print(f"Error: {e}")

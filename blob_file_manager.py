"""
Blob File Manager for Trinity Online
Handles all Azure Blob Storage file operations and provides clean interface for Streamlit UI.
"""

import os
from typing import List, Dict, Optional, Tuple
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError
from dotenv import load_dotenv


class BlobFileManager:
    """Centralized manager for Azure Blob Storage file operations."""
    
    def __init__(self):
        """Initialize the blob file manager with Azure configuration."""
        load_dotenv()
        
        # Azure Blob Storage configuration
        self.connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
        self.container_name = os.getenv("AZURE_BLOB_CONTAINER")
        
        # Connection status
        self.is_configured = bool(self.connection_string and self.container_name)
        self.service_client = None
        
        if self.is_configured:
            try:
                self.service_client = BlobServiceClient.from_connection_string(self.connection_string)
                self._test_connection()
            except Exception as e:
                print(f"Warning: Failed to initialize blob service client: {e}")
                self.is_configured = False

    def _test_connection(self) -> bool:
        """Test the blob storage connection."""
        try:
            # Try to list containers to test connection
            list(self.service_client.list_containers(max_results=1))
            return True
        except Exception:
            return False

    def get_connection_status(self) -> Dict[str, any]:
        """Get detailed connection status information."""
        return {
            'is_configured': self.is_configured,
            'has_connection_string': bool(self.connection_string),
            'has_container_name': bool(self.container_name),
            'container_name': self.container_name,
            'service_client_available': self.service_client is not None
        }

    def list_all_files(self) -> List[str]:
        """
        List all files in the blob storage container.
        
        Returns:
            List of file names in the container
        """
        if not self.is_configured or not self.service_client:
            return []
        
        try:
            blob_list = self.service_client.get_container_client(self.container_name).list_blobs()
            return [blob.name for blob in blob_list]
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def categorize_files(self) -> Dict[str, List[str]]:
        """
        Categorize files by type for better organization.
        
        Returns:
            Dictionary with file categories and lists of files
        """
        all_files = self.list_all_files()
        
        categories = {
            'excel': [],
            'pdf': [],
            'image': [],
            'other': []
        }
        
        for file in all_files:
            file_lower = file.lower()
            if file_lower.endswith(('.xlsx', '.xls')):
                categories['excel'].append(file)
            elif file_lower.endswith('.pdf'):
                categories['pdf'].append(file)
            elif file_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
                categories['image'].append(file)
            else:
                categories['other'].append(file)
        
        return categories

    def get_file_info(self, filename: str) -> Optional[Dict[str, any]]:
        """
        Get detailed information about a specific file.
        
        Args:
            filename: Name of the file to get info for
            
        Returns:
            Dictionary with file information or None if file not found
        """
        if not self.is_configured or not self.service_client:
            return None
        
        try:
            blob_client = self.service_client.get_blob_client(
                container=self.container_name, 
                blob=filename
            )
            properties = blob_client.get_blob_properties()
            
            return {
                'name': filename,
                'size': properties.size,
                'size_mb': round(properties.size / (1024 * 1024), 2),
                'last_modified': properties.last_modified,
                'content_type': properties.content_settings.content_type,
                'creation_time': properties.creation_time,
                'etag': properties.etag
            }
        except Exception as e:
            print(f"Error getting file info for {filename}: {e}")
            return None

    def download_file(self, filename: str) -> Optional[bytes]:
        """
        Download a file from blob storage as bytes.
        
        Args:
            filename: Name of the file to download
            
        Returns:
            File content as bytes or None if download failed
        """
        if not self.is_configured or not self.service_client:
            return None
        
        try:
            blob_client = self.service_client.get_blob_client(
                container=self.container_name, 
                blob=filename
            )
            return blob_client.download_blob().readall()
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return None

    def upload_file(self, filename: str, file_content: bytes, overwrite: bool = False) -> bool:
        """
        Upload a file to blob storage.
        
        Args:
            filename: Name for the file in blob storage
            file_content: File content as bytes
            overwrite: Whether to overwrite existing file
            
        Returns:
            True if upload successful, False otherwise
        """
        if not self.is_configured or not self.service_client:
            return False
        
        try:
            blob_client = self.service_client.get_blob_client(
                container=self.container_name, 
                blob=filename
            )
            blob_client.upload_blob(file_content, overwrite=overwrite)
            return True
        except Exception as e:
            print(f"Error uploading {filename}: {e}")
            return False

    def delete_file(self, filename: str) -> bool:
        """
        Delete a file from blob storage.
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        if not self.is_configured or not self.service_client:
            return False
        
        try:
            blob_client = self.service_client.get_blob_client(
                container=self.container_name, 
                blob=filename
            )
            blob_client.delete_blob()
            return True
        except Exception as e:
            print(f"Error deleting {filename}: {e}")
            return False

    def file_exists(self, filename: str) -> bool:
        """
        Check if a file exists in blob storage.
        
        Args:
            filename: Name of the file to check
            
        Returns:
            True if file exists, False otherwise
        """
        if not self.is_configured or not self.service_client:
            return False
        
        try:
            blob_client = self.service_client.get_blob_client(
                container=self.container_name, 
                blob=filename
            )
            blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False
        except Exception:
            return False

    def get_supported_files_for_processing(self) -> List[Dict[str, any]]:
        """
        Get list of files that can be processed by Trinity Online.
        
        Returns:
            List of dictionaries with file information for supported files
        """
        categories = self.categorize_files()
        supported_files = []
        
        # Add Excel files
        for filename in categories['excel']:
            file_info = self.get_file_info(filename)
            if file_info:
                file_info['category'] = 'Excel'
                file_info['icon'] = '📊'
                file_info['processor'] = 'Direct table extraction'
                supported_files.append(file_info)
        
        # Add PDF files
        for filename in categories['pdf']:
            file_info = self.get_file_info(filename)
            if file_info:
                file_info['category'] = 'PDF'
                file_info['icon'] = '📄'
                file_info['processor'] = 'OCR + table extraction'
                supported_files.append(file_info)
        
        # Add Image files
        for filename in categories['image']:
            file_info = self.get_file_info(filename)
            if file_info:
                file_info['category'] = 'Image'
                file_info['icon'] = '🖼️'
                file_info['processor'] = 'OCR + table extraction'
                supported_files.append(file_info)
        
        # Sort by last modified (newest first)
        supported_files.sort(key=lambda x: x.get('last_modified', ''), reverse=True)
        
        return supported_files

    def get_file_selection_data(self) -> Dict[str, any]:
        """
        Get comprehensive data for Streamlit file selection interface.
        
        Returns:
            Dictionary with all data needed for file selection UI
        """
        connection_status = self.get_connection_status()
        
        if not connection_status['is_configured']:
            return {
                'status': 'error',
                'message': 'Azure Blob Storage not configured',
                'files': [],
                'categories': {},
                'total_files': 0
            }
        
        categories = self.categorize_files()
        supported_files = self.get_supported_files_for_processing()
        
        return {
            'status': 'success',
            'connection': connection_status,
            'files': supported_files,
            'categories': categories,
            'total_files': sum(len(files) for files in categories.values()),
            'supported_count': len(supported_files),
            'container_name': self.container_name
        }


# Convenience function for easy import
def get_blob_file_manager() -> BlobFileManager:
    """Get a configured BlobFileManager instance."""
    return BlobFileManager()


# Example usage and testing
if __name__ == "__main__":
    print("🔍 TESTING BLOB FILE MANAGER")
    print("=" * 40)
    
    manager = get_blob_file_manager()
    status = manager.get_connection_status()
    
    print(f"Configuration Status: {'✅ Connected' if status['is_configured'] else '❌ Not configured'}")
    print(f"Container: {status['container_name']}")
    
    if status['is_configured']:
        selection_data = manager.get_file_selection_data()
        print(f"\nFiles Summary:")
        print(f"  Total files: {selection_data['total_files']}")
        print(f"  Supported files: {selection_data['supported_count']}")
        
        categories = selection_data['categories']
        print(f"\nBy Category:")
        print(f"  📊 Excel: {len(categories['excel'])}")
        print(f"  📄 PDF: {len(categories['pdf'])}")
        print(f"  🖼️ Image: {len(categories['image'])}")
        print(f"  📁 Other: {len(categories['other'])}")
    else:
        print("\n❌ Please configure Azure Blob Storage in .env file")
